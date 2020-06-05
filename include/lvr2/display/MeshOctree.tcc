#include <stdio.h>
#include <cmath>

#include "lvr2/io/Timestamp.hpp"

// TODO reorder colors etc...

namespace lvr2{ 
    template <typename BaseVecT>
        MeshOctree<BaseVecT>::MeshOctree(float voxelSize, std::vector<size_t>& hashes, std::vector<BaseVecT>& centroids, BoundingBox<BaseVecT>& bb) :
            m_voxelSize(voxelSize),
            m_bbox(bb),
            numLeafs(0)
    {
        long offset = 0;
        size_t numChunks = std::floor(bb.getXSize()/voxelSize);

        std::cout << sizeof(ChunkLeaf) << std::endl;

        std::cout << "BB unadjusted " << bb << std::endl;
        // check if it is a power of 2
        // https://stackoverflow.com/questions/600293/how-to-check-if-a-number-is-a-power-of-2
        if(!((numChunks & (numChunks - 1)) == 0))
        {
           // adjust number 2 the next power of 2
           // https://stackoverflow.com/questions/466204/rounding-up-to-next-power-of-2
           numChunks--;
           numChunks |= numChunks >> 1;
           numChunks |= numChunks >> 2;
           numChunks |= numChunks >> 4;
           numChunks |= numChunks >> 8;
           numChunks |= numChunks >> 16;
           //numChunks |= numChunks >> 32;
           numChunks++;
        }

        std::cout << numChunks << std::endl;
        BaseVecT min = bb.getMin();
        min += (BaseVecT(1, 0, 0) * voxelSize) * numChunks;
        std::cout << "min " << min << std::endl;
        bb.expand(min);

        numChunks = std::floor(bb.getYSize()/voxelSize);
        std::cout << std::endl;
        std::cout << numChunks << std::endl;
        // check if it is a power of 2
        // https://stackoverflow.com/questions/600293/how-to-check-if-a-number-is-a-power-of-2
        if(!((numChunks & (numChunks - 1)) == 0))
        {
            std::cout << "NO power of 2" << std::endl;
           // adjust number 2 the next power of 2
           // https://stackoverflow.com/questions/466204/rounding-up-to-next-power-of-2
           numChunks--;
           numChunks |= (numChunks >> 1);
           numChunks |= (numChunks >> 2);
           numChunks |= (numChunks >> 4);
           numChunks |= (numChunks >> 8);
           numChunks |= (numChunks >> 16);
        //   numChunks |= numChunks >> 32;
           numChunks++;
        }

        std::cout << numChunks << std::endl;
        min = bb.getMin();
        min += BaseVecT(0, 1, 0) * voxelSize * numChunks;
        bb.expand(min);
        std::cout << "min " << min << std::endl;

        numChunks = std::floor(bb.getZSize()/voxelSize);
        std::cout << "prev " << numChunks << std::endl;
        // check if it is a power of 2
        // https://stackoverflow.com/questions/600293/how-to-check-if-a-number-is-a-power-of-2
        if(!((numChunks & (numChunks - 1)) == 0))
        {
            std::cout << "no power of 2" << std::endl;
           // adjust number 2 the next power of 2
           // https://stackoverflow.com/questions/466204/rounding-up-to-next-power-of-2
           numChunks--;
           numChunks |= numChunks >> 1;
           numChunks |= numChunks >> 2;
           numChunks |= numChunks >> 4;
           numChunks |= numChunks >> 8;
           numChunks |= numChunks >> 16;
           numChunks |= numChunks >> 32;
           numChunks++;
        }
        else
        {
            std::cout << "multiply " << std::endl;
        }

        numChunks *= 2;
        std::cout << numChunks << std::endl;
        min = bb.getMin();
        min += BaseVecT(0, 0, 1) * voxelSize * numChunks;
        bb.expand(min);
        std::cout << "min " << min << std::endl;

        std::cout << "BB adjusted " << bb << " voxelSize " << voxelSize << std::endl;
        m_bbox = bb;



        std::cout << lvr2::timestamp << "Start building octree with voxelsize " << m_voxelSize << std::endl;
        std::cout << lvr2::timestamp << hashes.size() << std::endl;
        m_root = reinterpret_cast<BOct*>(m_mem.alloc<BOct>(1, offset));
        m_root = (BOct*)((unsigned char*) m_root + buildTree(m_root, hashes, centroids, m_bbox));
//        buildTree(m_root, hashes, centroids, m_bbox);
        std::cout << lvr2::timestamp << numLeafs << std::endl;
    }

    template  <typename BaseVecT>
        void MeshOctree<BaseVecT>::getBBoxes(const BoundingBox<BaseVecT>& bbox, BoundingBox<BaseVecT>* boxes)
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
        unsigned char MeshOctree<BaseVecT>::getIndex(const BaseVecT& point, const BoundingBox<BaseVecT>& bbox)
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
        inline void MeshOctree<BaseVecT>::link(BOct* parent, T* child)
        {
            parent->m_child =(long)((unsigned char*)child - (unsigned char*)parent);
        }

    template <typename BaseVecT>
        template <typename T>
        inline T* MeshOctree<BaseVecT>::getChildPtr(BOct* parent)
        {
            return reinterpret_cast<T*>((unsigned char*)parent + (long long) parent->m_child);
        }

    //  template <typename BaseVecT>
    //    template <typename PtrT>
    //    void MeshOctree<BaseVecT>::sortPC(size_t start, size_t size, const BoundingBox<BaseVecT>& bbox, size_t bucket_sizes[8])
    //    {
    //      // TODO template this properly
    //      PtrT ptr[8];
    //      PtrT beg[8];
    //
    //      beg[0] = ptr[0] = &m_points[start];
    ////      beg[0] = ptr[0] = p;
    //
    //      //creating pointers for bucket starts.
    //      for(int j = 1; j < 8; ++j)
    //      {
    //        // pointing to first position of bucket.
    //        beg[j] = ptr[j] = ptr[j - 1] + bucket_sizes[j - 1];
    //      }
    //
    //      size_t end = start + size;
    //            //size_t full = ptr[1] - &pts[0];
    //      for(size_t i = start; i < end; )
    //      {
    //        int index = getIndex(m_points[i], bbox);
    //        if(ptr[index] == &(m_points[end]))
    //        {
    //          std::cout << "This should have never happened." << std::endl;
    //          std::cout << (int) index << std::endl;
    //        }
    //
    //        if((beg[index] <= &m_points[i]) && (&m_points[i] < ptr[index]))
    //        {
    //          i = ptr[index] - &m_points[0];
    //          continue;
    //        }
    //
    //        
    //        if(ptr[index] == &m_points[i])
    //        {
    //          // is already in correct bucket 
    //          if(ptr[index] < &m_points[end - 1])
    //            ptr[index]++; 
    //
    //          i++;
    //        }
    //        else
    //        {
    //          // TODO 
    //          // We somehow need 2 temporary variables. Otherwise it won't work with the proxy(Ptr)
    //
    //          // advance bucket pointer if current element is correct.
    //          BaseVecT tmp = *ptr[index];
    //          while(getIndex(tmp, bbox) == index)
    //          {
    //            if(ptr[index] < &(m_points[end - 1]))
    //              ptr[index]++;
    //
    //            tmp = *ptr[index];
    //          }
    //          BaseVecT aux = m_points[i];
    //          *(ptr[index]) = aux;
    //          m_points[i] = tmp;
    //          if(ptr[index] < &(m_points[end - 1]))
    //            ptr[index]++;
    //        }
    //
    //      }
    //    }

    template <typename BaseVecT>
        long MeshOctree<BaseVecT>::buildTree(BOct* oct, std::vector<size_t>& hashes, std::vector<BaseVecT>& centroids, const BoundingBox<BaseVecT>& bbox)
        {
            if(centroids.empty())
            {
                return 0;
            }
            BoundingBox<BaseVecT> boxes[8];
            getBBoxes(bbox, boxes);
            std::vector<size_t>   c_hashes[8];
            std::vector<BaseVecT> c_centroids[8];
            size_t numChildren = 0;

            // LEAF
            if(bbox.getXSize()/2.0 <= m_voxelSize)
            {
                for(size_t i = 0; i < centroids.size(); ++i)
                {
                    unsigned char index = getIndex(centroids[i], bbox);
                    c_centroids[index].push_back(centroids[i]);
                    c_hashes[index].push_back(hashes[i]);

                    if(!(oct->m_leaf & (1 << index)))
                    {
                        oct->m_leaf |= (1 << index);
                        numChildren++;
                    }
                }
                hashes.clear();
                std::vector<size_t>().swap(hashes);
                centroids.clear();
                std::vector<BaseVecT >().swap(centroids);

                long offset = 0;
                numLeafs += numChildren;
                ChunkLeaf* leaves = reinterpret_cast<ChunkLeaf*>(m_mem.alloc<ChunkLeaf>(numChildren, offset));
//                ChunkLeaf* leaves = new Leaf[numChildren];
                if(offset)
                {
                    std::cout << "THIS SHOULD NEVER HAPPEN" << std::endl;
                    oct = (BOct*) ((unsigned char*)oct + offset);
                }

                link(oct, leaves);
            
                int cnt = 0;
                for(unsigned i = 0; i < 8; ++i)
                {
                    if(oct->m_leaf & (1 << i))
                    {
                        // NEED TO INITIALIZE THIS VECTORS...
                        leaves[cnt].m_centroids = std::vector<BaseVecT>();
                        leaves[cnt].m_hashes = std::vector<size_t>();
                        if((void*)(&(leaves[cnt].m_centroids)) == (void*)(&(leaves[cnt].m_hashes)))
                        {
                            exit(1);
                        }
                        leaves[cnt].m_centroids.resize(c_centroids[i].size());
                        leaves[cnt].m_hashes.resize(c_hashes[i].size());
                        if(c_hashes[i].size() != c_centroids[i].size())
                        {
                            std::cout << "SIZES DIFFER!!!" << std::endl;
                        }

                        for(size_t j = 0 ; j < c_hashes[i].size(); ++j)
                        {
                            // SHOULD ONLY BE ONE!!
                            
                            //size_t hash = (c_hashes[i][j]);
                            //BaseVecT cent = (c_centroids[i][j]);
                            //leaves[cnt].m_hashes[j]    = hash;
                            //leaves[cnt].m_centroids[j] = cent;
                            leaves[cnt].m_hashes[j] = c_hashes[i][j];
                            leaves[cnt].m_centroids[j] = c_centroids[i][j];
//                            std::cout << j << std::endl;
                        }
                        cnt++;
                    }
                }
                return offset;
            }

            for(size_t i = 0; i < centroids.size(); ++i)
            {
                unsigned char index = getIndex(centroids[i], bbox);
                c_centroids[index].push_back(centroids[i]);
                c_hashes[index].push_back(hashes[i]);
                if(!(oct->m_valid & (1 << index)))
                {
                    //std::cout << "setting new valid " << index << std::endl;
                    oct->m_valid |= (1 << index);
                    numChildren++;
                }
            }

            //sortPC(&m_points[start], start, size, bbox, octSizes);
            //      sortPC<decltype(&(m_points[start]))>(start, size, bbox, octSizes);
            // force reallocation to clear vec
            //      pts.clear();
            //      std::vector<BaseVecT >().swap(pts);

            hashes.clear();
            std::vector<size_t>().swap(hashes);
            centroids.clear();
            std::vector<BaseVecT >().swap(centroids);

            long offset = 0;
            BOct* newOct = reinterpret_cast<BOct*>(m_mem.alloc<BOct>(numChildren, offset));
            //BOct* newOct = new BOct[numChildren];

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
                    sub_offset += buildTree(&newOct[cnt++], c_hashes[i], c_centroids[i], boxes[i]);
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
    //    void MeshOctree<BaseVecT>::writeLeaf(ChunkLeaf* leaf, unsigned char index)
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
    //    void MeshOctree<BaseVecT>::colorAndWrite(BOct* oct, unsigned char index)
    //    {
    //
    //      unsigned char cnt = 0;
    //      if(oct->m_leaf)
    //      {
    //        ChunkLeaf* child = getChildPtr<Leaf>(oct);
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
    //    void MeshOctree<BaseVecT>::colorAndWrite(BOct* oct)
    //    {
    //      static int depth = 0;
    //      unsigned char cnt = 0;
    //      if(oct->m_leaf)
    //      {
    //        ChunkLeaf* child = getChildPtr<Leaf>(oct);
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
        void MeshOctree<BaseVecT>::getHashes(BOct* oct, std::vector<BaseVecT>& indices, std::vector<size_t>& hashes)
        {
            unsigned char cnt = 0;
            if(!oct)
                return;
            if(oct->m_leaf)
            {
                ChunkLeaf* leaf = getChildPtr<ChunkLeaf>(oct);
                for(unsigned char i = 0; i < 8; ++i)
                {
                    if(oct->m_leaf & (1 << i))
                    {
                        ChunkLeaf* l = leaf + cnt;
                        for(size_t j = 0; j < l->m_hashes.size(); ++j)
                        {
                            indices.push_back(l->m_centroids[j]);
                            hashes.push_back(l->m_hashes[j]);
                        }

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
                        getHashes(child + cnt++, indices, hashes);
                    }
                }
            }
        }


    template <typename BaseVecT>
        void MeshOctree<BaseVecT>::intersect(double planes[24], std::vector<BaseVecT>& indices, std::vector<size_t>& hashes)
        {
//            normalizePlanes(planes);
            intersect(m_root, m_bbox, planes, indices, hashes);
        }

    template <typename BaseVecT>
        void MeshOctree<BaseVecT>::intersect(ChunkLeaf* leaf, const BoundingBox<BaseVecT>& bbox, double planes[24], std::vector<BaseVecT >& indices, std::vector<size_t>& hashes)
        {
            if(!leaf)
                return;

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
            for(size_t j = 0; j < leaf->m_hashes.size(); ++j)
            {
                indices.push_back(leaf->m_centroids[j]);
                hashes.push_back(leaf->m_hashes[j]);
            }
        }

    template <typename BaseVecT>
        void MeshOctree<BaseVecT>::intersect(BOct* oct, const BoundingBox<BaseVecT>& bbox, double planes[24], std::vector<BaseVecT >& indices, std::vector<size_t>& hashes)
        {

            bool inlier = true;

            if(!oct)
                return;

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
                return getHashes(oct, indices, hashes);
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
//                        std::cout << cnt << std::endl;
                        intersect(getChildPtr<ChunkLeaf>(oct) + cnt++, bboxes[i], planes, indices, hashes);
                    }
                }
            }
            else
            {
                for(unsigned char i = 0; i < 8; ++i)
                {
                    if(oct->m_valid & (1 << i))
                    {
                        intersect(getChildPtr<BOct>(oct) + cnt++, bboxes[i], planes, indices, hashes);
                    }
                }
            }
        }
}
