#include <stdio.h>
#include <cmath>

#include "lvr2/io/Timestamp.hpp"

// TODO reorder colors etc...

namespace lvr2{ 
  template <typename BaseVecT>
    PointOctree<BaseVecT>::PointOctree(PointBufferPtr& points, int depth) : m_points(*(points->getFloatChannel("points")))
    {

//      FloatChannelOptional pts_channel = points->getFloatChannel("points");
//      m_points = *pts_channel;

      // initializ min max for bounding box
      BaseVecT p = m_points[0];
      float minX = p.x;
      float minY = p.y;
      float minZ = p.z;
      float maxX = p.x;
      float maxY = p.y;
      float maxZ = p.z;



      for(int i = 0; i < points->numPoints() - 1; ++i)
      {
        p = m_points[i];
        minX = std::min(minX, p.x);
        minY = std::min(minY, p.y);
        minZ = std::min(minZ, p.z);

        maxX = std::max(maxX, p.x);
        maxY = std::max(maxY, p.y);
        maxZ = std::max(maxZ, p.z);
      }

      // make sure all points are inliers
      minX -= 1.0; 
      minY -= 1.0; 
      minZ -= 1.0; 
      maxX += 1.0; 
      maxY += 1.0; 
      maxZ += 1.0; 

      // make it square, there has to be a more elegant solution.
      float min = std::min(minX, std::min(minY, minZ));
      float max = std::max(maxX, std::max(maxY, maxZ));

      BaseVecT v1(min, min, min);
      BaseVecT v2(max, max, max);

      m_bbox = BoundingBox<BaseVecT>(v1, v2);

//      int depth = std::ceil(std::log2(m_bbox.getLongestSide()/m_voxelSize));
      m_voxelSize = m_bbox.getXSize() / std::pow(2, depth);

      long offset = 0;
      m_root = reinterpret_cast<BOct*>(m_mem.alloc<BOct>(1, offset));

      //std::cout << msg << std::endl;
      //if(offset)
      //{
      //  std::cout << "offset" << std::endl;
      //}

//      std::vector<BaseVecT > pts = points->getPointBufferReference();
      std::cout << m_bbox << std::endl;

      std::cout << lvr2::timestamp << "Start building octree with voxelsize " << m_voxelSize << std::endl;
      m_root = (BOct*)((unsigned char*) m_root + buildTree(m_root, 0, points->numPoints(), m_bbox));

      std::cout << lvr2::timestamp << "generating genDisplayLists " << std::endl;
      genDisplayLists();

      std::cout << lvr2::timestamp << "generating genDisplayLists done" << std::endl;
      std::cout << lvr2::timestamp << "Octree rdy " << std::endl;

//        m_points.clear();
//        std::vector<BaseVecT >().swap(m_points);

      //  colorAndWrite(m_root);

    }

  template  <typename BaseVecT>
    void PointOctree<BaseVecT>::getBBoxes(const BoundingBox<BaseVecT>& bbox, BoundingBox<BaseVecT>* boxes)
    {

      BaseVecT centroid = bbox.getCentroid();

      // square bbox
      auto size = bbox.getXSize();

      for(int i = 0; i < 8; ++i)
      {
        BaseVecT bboxLowerLeft = centroid;
        BaseVecT bboxTopRight = centroid;

        // top
        if(i & 4)
        {
          bboxTopRight.x += size / 2.0;
        }
        else
        {
          bboxLowerLeft.x -= size / 2.0;
        }

        if(i & 2)
        {
          bboxTopRight.y  += size / 2.0;
        }
        else
        {
          bboxLowerLeft.y -= size / 2.0;
        }

        if(i & 1)
        {
          bboxTopRight.z += size / 2.0;
        }
        else
        {
          bboxLowerLeft.z -= size / 2.0;
        }

        boxes[i] = BoundingBox<BaseVecT>(bboxLowerLeft, bboxTopRight);
      }
    }

  template <typename BaseVecT> 
    unsigned char PointOctree<BaseVecT>::getIndex(const BaseVecT& point, const BoundingBox<BaseVecT>& bbox)
    {
      BaseVecT centroid = bbox.getCentroid();
      unsigned char index = 0;


      // TODO i think this is not consistent. Coordinate system
      // 
      // y
      // |   x
      // |  /       ???
      // | /       
      // |/_____ z
      if(point.x > centroid.x)
      {
        index += 4;
      }
      // "top"
      if(point.y > centroid.y)
      {
        index += 2;
      }
      // "right"
      if(point.z > centroid.z)
      {
        index += 1;
      }

      return index;
    }

  template <typename BaseVecT>
    template <typename T>
    inline void PointOctree<BaseVecT>::link(BOct* parent, T* child)
    {
      parent->m_child =(long)((unsigned char*)child - (unsigned char*)parent);
    }

  template <typename BaseVecT>
    template <typename T>
    inline T* PointOctree<BaseVecT>::getChildPtr(BOct* parent)
    {
      return reinterpret_cast<T*>((unsigned char*)parent + parent->m_child);
    }

  template <typename BaseVecT>
    template <typename PtrT>
    void PointOctree<BaseVecT>::sortPC(size_t start, size_t size, const BoundingBox<BaseVecT>& bbox, size_t bucket_sizes[8])
    {
      // TODO template this properly
      PtrT ptr[8];
      PtrT beg[8];

      beg[0] = ptr[0] = &m_points[start];
//      beg[0] = ptr[0] = p;

      //creating pointers for bucket starts.
      for(int j = 1; j < 8; ++j)
      {
        // pointing to first position of bucket.
        beg[j] = ptr[j] = ptr[j - 1] + bucket_sizes[j - 1];
      }

      size_t end = start + size;
            //size_t full = ptr[1] - &pts[0];
      for(size_t i = start; i < end; )
      {
        int index = getIndex(m_points[i], bbox);
        if(ptr[index] == &(m_points[end]))
        {
          std::cout << "This should have never happened." << std::endl;
          std::cout << (int) index << std::endl;
        }

        if((beg[index] <= &m_points[i]) && (&m_points[i] < ptr[index]))
        {
          i = ptr[index] - &m_points[0];
          continue;
        }

        
        if(ptr[index] == &m_points[i])
        {
          // is already in correct bucket 
          if(ptr[index] < &m_points[end - 1])
            ptr[index]++; 

          i++;
        }
        else
        {
          // TODO 
          // We somehow need 2 temporary variables. Otherwise it won't work with the proxy(Ptr)

          // advance bucket pointer if current element is correct.
          BaseVecT tmp = *ptr[index];
          while(getIndex(tmp, bbox) == index)
          {
            if(ptr[index] < &(m_points[end - 1]))
              ptr[index]++;

            tmp = *ptr[index];
          }
          BaseVecT aux = m_points[i];
          *(ptr[index]) = aux;
          m_points[i] = tmp;
          if(ptr[index] < &(m_points[end - 1]))
            ptr[index]++;
        }

      }
    }

  template <typename BaseVecT>
    long PointOctree<BaseVecT>::buildTree(BOct* oct, size_t start, size_t size, const BoundingBox<BaseVecT>& bbox)
    {
      // TODO did i need this.
     // if(pts.empty())
     // {
     //   return 0;
     // }
      BoundingBox<BaseVecT> boxes[8];
      getBBoxes(bbox, boxes);

//      std::vector<BaseVecT > octPoints[8];

      size_t octSizes[8] = {0, 0, 0, 0, 0, 0, 0, 0};

      //std::cout << "bbox size " << bbox.getXSize() << std::endl;
      if(bbox.getXSize()/2.0 <= m_voxelSize)
      {
        //std::cout << "leaf" << std::endl;
        int numChildren = 0;
        for(size_t i = start; i < (start + size); ++i)
        {
          unsigned char index = getIndex(m_points[i], bbox);
//          octPoints[index].push_back(point);
          octSizes[index]++;
          //printf("Address %p\n", oct);
          if(!(oct->m_leaf & (1 << index)))
          {
            oct->m_leaf |= (1 << index);
            numChildren++;
          }
        }
        
//        sortPC(&m_points[start], start, size, bbox, octSizes);
        sortPC<decltype(&(m_points[start]))>(start, size, bbox, octSizes);
     //   sortPC(start, size, bbox, octSizes);

        // force reallocation to clear vec
//        pts.clear();
//        std::vector<BaseVecT >().swap(pts);

        long offset = 0;

        //        std::cout << "no leafs. " << numChildren << std::endl;
        Leaf* leaves = reinterpret_cast<Leaf*>(m_mem.alloc<Leaf>(numChildren, offset));

        if(offset)
        {
          oct = (BOct*) ((unsigned char*)oct + offset);
        }

        link(oct, leaves);

        int cnt = 0;
        for(unsigned i = 0; i < 8; ++i)
        {
          if(oct->m_leaf & (1 << i))
          {
            leaves[cnt].m_start = start;
            leaves[cnt].m_size = octSizes[i];
            cnt++;
            start += octSizes[i];
//            leaves[cnt].m_size = octPoints[i].size();
//            if(m_points.empty())
//            {
//              leaves[cnt].m_start = 0;
//            }
//            else
//            {
//              leaves[cnt].m_start = m_points.end() - m_points.begin();
//            }
//            m_points.insert(m_points.end(), octPoints[i].begin(), octPoints[i].end());
//            cnt++;
//            // clear and force reallocation
//            octPoints[i].clear();
//            std::vector<BaseVecT >().swap(octPoints[i]);
          }
        }
        //std::cout << "return offset " << offset << std::endl;
        return offset;
      }

      int numChildren = 0;
      for(size_t i = start; i < (start + size); ++i)
      {
        unsigned char index = getIndex(m_points[i], bbox);
//        octPoints[index].push_back(point);
        octSizes[index]++;
        if(!(oct->m_valid & (1 << index)))
        {
          //std::cout << "setting new valid " << index << std::endl;
          oct->m_valid |= (1 << index);
          numChildren++;
        }
      }

      //sortPC(&m_points[start], start, size, bbox, octSizes);
      sortPC<decltype(&(m_points[start]))>(start, size, bbox, octSizes);
      // force reallocation to clear vec
//      pts.clear();
//      std::vector<BaseVecT >().swap(pts);

      long offset = 0;

      BOct* newOct = reinterpret_cast<BOct*>(m_mem.alloc<BOct>(numChildren, offset));

      //      printf("Address %p\n", newOct);
      // realloc in blockalloc may cause an address change.
      if(offset)
      {

        oct = (BOct*) ((unsigned char*)oct + offset);
      }


      link(oct, newOct);


      unsigned char cnt = 0;
      for(unsigned char i = 0; i < 8; ++i)
      {
        long sub_offset = 0;
        if(oct->m_valid & (1 << i))
        {
          sub_offset += buildTree(&newOct[cnt++], start, octSizes[i], boxes[i]);
          start += octSizes[i];
          if(sub_offset)
          {
            newOct = (BOct*) ((unsigned char*)newOct + sub_offset);
            oct = (BOct*) ((unsigned char*)oct + sub_offset);
          }
          offset += sub_offset;
        }
      }

      return offset;

      // due to reallocation we might need to fix the root and the current oct address
      // higher level octants are not effected because they are already linked.
      // BULLSHIT we allocate in lower levels and then go to the next octant

    }

//  template <typename BaseVecT>
//    void PointOctree<BaseVecT>::writeLeaf(Leaf* leaf, unsigned char index)
//    {
//      //std::cout << leaf->m_start << " " << leaf->m_size << std::endl;
//      for(unsigned int i = leaf->m_start; i < (leaf->m_start + leaf->m_size); ++i)
//      {
//        auto point = m_points[i];
//
//        std::cout << point.x << " " 
//          << point.y << " "
//          << point.z << " "
//          << (int)pow(2, index) << " "
//          << (int)pow(2, index) << " " 
//          << (int)pow(2, index) << " "
//          << std::endl;
//      }
//    }
//
//  template <typename BaseVecT>
//    void PointOctree<BaseVecT>::colorAndWrite(BOct* oct, unsigned char index)
//    {
//
//      unsigned char cnt = 0;
//      if(oct->m_leaf)
//      {
//        Leaf* child = getChildPtr<Leaf>(oct);
//        for(unsigned char i = 0; i < 8; ++i)
//        {
//          if(oct->m_leaf & (1 << i))
//          {
//            writeLeaf(child + cnt, i);
//            cnt++;
//          }
//        }
//      }
//      else
//      {
//        BOct* child = getChildPtr<BOct>(oct);
//        for(unsigned char i = 0; i < 8; ++i)
//        {
//          if(oct->m_valid & (1 << i))
//          {
//            colorAndWrite(child + cnt, index);
//            cnt++;
//          }
//        }
//      }
//
//    }
//
//  template <typename BaseVecT>
//    void PointOctree<BaseVecT>::colorAndWrite(BOct* oct)
//    {
//      static int depth = 0;
//      unsigned char cnt = 0;
//      if(oct->m_leaf)
//      {
//        Leaf* child = getChildPtr<Leaf>(oct);
//        for(unsigned char i = 0; i < 8; ++i)
//        {
//          if(oct->m_leaf & (1 << i))
//          {
//            writeLeaf(child + cnt, i);
//            cnt++;
//          }
//        }
//      }
//      else
//      {
//        BOct* child = getChildPtr<BOct>(oct);
//        for(unsigned char i = 0; i < 8; ++i)
//        {
//          if(oct->m_valid & (1 << i))
//          {
//            if(oct != m_root)
//            {
//              colorAndWrite(child + cnt, i);
//              cnt++;
//            }
//            else
//            {
//              colorAndWrite(child + cnt);
//              cnt++;
//              depth++;
//            }
//          }
//        }
//      }
//    }

  template <typename BaseVecT>
    void PointOctree<BaseVecT>::getPoints(BOct* oct, std::vector<unsigned int>& indices)
    {
      unsigned char cnt = 0;
      if(oct->m_leaf)
      {
        Leaf* leaf = getChildPtr<Leaf>(oct);
        for(unsigned char i = 0; i < 8; ++i)
        {
          if(oct->m_leaf & (1 << i))
          {
            indices.push_back(leaf[cnt].m_listIndex + m_lod);
            
            //if(!m_lod)
            //{
            //  auto start = m_points.begin() + leaf[cnt].m_start;
            //  pts.insert(pts.end(), start, start + leaf[cnt].m_size);
            //}
            //else
            //{
            //  for(int i = 0; i < leaf[cnt].m_size; ++i)
            //  {
            //    if((i % m_lod) == 0)
            //    {
            //      pts.push_back(m_points[leaf[cnt].m_start + i]);
            //    }
            //  }
            //}
            cnt++;
          }
        }
      }
      else
      {
        BOct* child = getChildPtr<BOct>(oct);
        for(unsigned char i = 0; i < 8; ++i)
        {
          if(oct->m_valid & (1 << i))
          {
            getPoints(child + cnt++, indices);
          }
        }
      }
    }

  template <typename BaseVecT>
    void PointOctree<BaseVecT>::genDisplayLists(Leaf* leaf)
    {
      leaf->m_listIndex = glGenLists(5);
      for(int i = 0; i < 5; ++i)
      {
        glNewList(leaf->m_listIndex + i, GL_COMPILE);
        glBegin(GL_POINTS);

        size_t mod = 1;
        if(i == 9)
        {
          if(8 < leaf->m_size)
            mod = leaf->m_size / 8;
        }
        else if(i == 8)
        {
          if(16 < leaf->m_size)
            mod = leaf->m_size / 16;
        }
        else if(i == 7)
        {
          if(32 < leaf->m_size)
            mod = leaf->m_size / 32;
        }
        else if(i == 6)
        {
          if(64 < leaf->m_size)
            mod = leaf->m_size / 64;
        }
        else if(i == 5)
        {
          if(128 < leaf->m_size)
            mod = leaf->m_size / 128;
        }
        else if(i == 4)
        {
          if(256 < leaf->m_size)
            mod = leaf->m_size / 256;
        }
         else if(i == 3)
        {
          if(512 < leaf->m_size)
            mod = leaf->m_size / 512;
        }
        else if(i == 2)
        {
          if(1024 < leaf->m_size)
            mod = leaf->m_size / 1024;
        }
        else if(i == 1)
        {
          if(2048 < leaf->m_size)
            mod = leaf->m_size / 2048;
        }
 
 
        
        for(size_t j = leaf->m_start;
            j < (leaf->m_start + leaf->m_size);
            ++j)
        {
          if(j % mod)
            continue;

          glColor3f(255.0f, 255.0f, 255.0f);
          BaseVecT p = m_points[j];
          glVertex3f(p.x,
                     p.y,
                     p.z);
          }
        glEnd();
        glEndList();
      }
    }


  template <typename BaseVecT>
    void PointOctree<BaseVecT>::genDisplayLists(BOct* oct)
    {
      unsigned char cnt = 0;
      if(oct->m_leaf)
      {
        Leaf* leaf = getChildPtr<Leaf>(oct);
        for(unsigned char i = 0; i < 8; ++i)
        {
          if(oct->m_leaf & (1 << i))
          {
            genDisplayLists(leaf + cnt++);
          }
        }
      }
      else
      {
        BOct* child = getChildPtr<BOct>(oct);
        for(unsigned char i = 0; i < 8; ++i)
        {
          if(oct->m_valid & (1 << i))
          {
            genDisplayLists(child + cnt++);
          }
        }
      }
    }



  template <typename BaseVecT>
    void PointOctree<BaseVecT>::intersect(double planes[24], std::vector<unsigned int>& indices)
    {
//      normalizePlanes(planes);
      intersect(m_root, m_bbox, planes, indices);
    }

  template <typename BaseVecT>
    void PointOctree<BaseVecT>::intersect(Leaf* leaf, const BoundingBox<BaseVecT>& bbox, double planes[24], std::vector<unsigned int >& indices)
    {

      for(unsigned char i = 0; i < 6; ++i)
      {
        BaseVecT octMin = bbox.getMin();
        BaseVecT octMax = bbox.getMax();
        BaseVecT pVertex = octMin;

        if(planes[i * 4 + 0] >= 0)
        {
            pVertex.x = octMax.x;
        }
        if(planes[i * 4 + 1] >= 0)
        {
            pVertex.y = octMax.y;
        }

        if(planes[i * 4 + 2] >= 0)
        {
            pVertex.z = octMax.z;
        }

        double distance;
        // get distance pVertex. hessian 
        distance = planes[i * 4 + 0] * pVertex.x +
                   planes[i * 4 + 1] * pVertex.y +
                   planes[i * 4 + 2] * pVertex.z +
                   planes[i * 4 + 3];

        // outlier.
        if(distance < 0)
          return;
      }

      // if leaf is intersected it is not culled.
      indices.push_back(leaf->m_listIndex + m_lod);
    }

  template <typename BaseVecT>
    void PointOctree<BaseVecT>::intersect(BOct* oct, const BoundingBox<BaseVecT>& bbox, double planes[24], std::vector<unsigned int >& indices)
    {

      bool inlier = true;

      for(unsigned char i = 0; i < 6; ++i)
      {
        BaseVecT octMin = bbox.getMin();
        BaseVecT octMax = bbox.getMax();
        BaseVecT pVertex = octMin;
        BaseVecT nVertex = octMax;

        if(planes[i * 4 + 0] >= 0)
        {
            pVertex.x = octMax.x;
            nVertex.x = octMin.x;
        }
        if(planes[i * 4 + 1] >= 0)
        {
            pVertex.y = octMax.y;
            nVertex.y = octMin.y;
        }

        if(planes[i* 4 + 2] >= 0)
        {
            pVertex.z = octMax.z;
            nVertex.z = octMin.z;
        }

        double distance;

        // get distance pVertex. hessian 
        distance = planes[i * 4 + 0] * pVertex.x +
                   planes[i * 4 + 1] * pVertex.y +
                   planes[i * 4 + 2] * pVertex.z +
                   planes[i * 4 + 3];

        // outlier.
        if(distance < 0)
          return;

        // distance to nVertex
        distance = planes[i * 4 + 0] * nVertex.x +
                   planes[i * 4 + 1] * nVertex.y +
                   planes[i * 4 + 2] * nVertex.z +
                   planes[i * 4 + 3];

        if(distance < 0)
          inlier = false;
      }

      if(inlier)
      {
        return getPoints(oct, indices);
      }

      BoundingBox<BaseVecT> bboxes[8];
      getBBoxes(bbox, bboxes);

      unsigned char cnt = 0;
      if(oct->m_leaf)
      {
        for(unsigned char i = 0; i < 8; ++i)
        {
          if(oct->m_leaf & (1 << i))
          {
            intersect(getChildPtr<Leaf>(oct) + cnt++, bboxes[i], planes, indices);
          }
        }
      }
      else
      {
        for(unsigned char i = 0; i < 8; ++i)
        {
          if(oct->m_valid & (1 << i))
          {
            intersect(getChildPtr<BOct>(oct) + cnt++, bboxes[i], planes, indices);
          }
        }
      }
    }

}
