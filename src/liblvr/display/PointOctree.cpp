#include <lvr2/display/PointOctree.hpp>
#include <algorithm>
#include <cmath>

namespace lvr2
{
  using Vec = BaseVector<float>;

  PointOctree::PointOctree(PointBufferPtr<Vec >& pts, int voxelSize)
  {
    m_voxelSize = voxelSize;

    // initializ min max for bounding box
    float minX = pts->getPoint(0).x;
    float minY = pts->getPoint(0).y;
    float minZ = pts->getPoint(0).z;
    float maxX = pts->getPoint(0).x;
    float maxY = pts->getPoint(0).y;
    float maxZ = pts->getPoint(0).z;

    //    BoundingBox<Vec> bb;
    for(int i = 0; i < pts->getNumPoints(); ++i)
    {
      auto p = pts->getPoint(i);
      minX = std::min(minX, p.x);
      minY = std::min(minY, p.y);
      minZ = std::min(minZ, p.z);

      maxX = std::max(maxX, p.x);
      maxY = std::max(maxY, p.y);
      maxZ = std::max(maxZ, p.z);
    }

    // be safe all points are inliers
    minX -= 1.0; 
    minY -= 1.0; 
    minZ -= 1.0; 
    maxX += 1.0; 
    maxY += 1.0; 
    maxZ += 1.0; 

    // make it square, there has to be a more elegant solution.
    float min = std::min(minX, std::min(minY, minZ));
    float max = std::max(maxX, std::max(maxY, maxZ));

    Point<Vec> v1(min, min, min);
    Point<Vec> v2(max, max, max);

    m_bbox = BoundingBox<Vec>(v1, v2);

    // number of splits to get a resolution smaller than voxelSize
    int depth = std::ceil(std::log2(m_bbox.getLongestSide()/m_voxelSize));

    m_root = new BOct();

    for(int i = 0; i < pts->getNumPoints(); ++i)
    {
      insertPoint(pts->getPoint(i), m_root, m_bbox);
    }
  }
  
  int PointOctree::octantIndex(const Point<Vec >& point, const BoundingBox<Vec >& bbox)
  {
  
  }

  inline void PointOctree::insertPoint(const Point<Vec >& point, BOct* oct, const BoundingBox<Vec >& bbox)
  {   
    int index = 0;

    Point<Vec > centroid = bbox.getCentroid();

    Point<Vec > bboxLowerLeft = centroid;
    Point<Vec > bboxTopRight = centroid;

    // calculate "indices" of subtree and boundingbox 
    // "back"
    if(point.x > centroid.x)
    {
      bboxTopRight.x += bbox.getXSize()/2;
      index += 4;
    }
    else
    {
      bboxLowerLeft.x -= bbox.getXSize()/2;
    }
    // "top"
    if(point.y > centroid.y)
    {
      bboxTopRight.y += bbox.getYSize()/2;
      index += 2;
    }
    else
    {
      bboxLowerLeft.y -= bbox.getYSize()/2;
    }
    // "right"
    if(point.z > centroid.z)
    {
      bboxTopRight.z += bbox.getZSize()/2;
      index += 1;
    }
    else
    {
      bboxLowerLeft.z -= bbox.getZSize()/2;
    }

    // next is leaf
    if(bbox.getXSize()/2 <= m_voxelSize) //bbox is square
    {
      // simply set it, if it is already set nothing changes
      oct->m_leaf = oct->m_leaf | (1 << index);
      // recursion anchor
      // now we have build the tree structure without the leaves
      return;
    }

    int position = 0;
    // create new octant and find position
    if(!((index >> oct->m_valid) & 1))
    {
      unsigned long tmp = oct->m_child;
      // adjust leaf bit mask
      oct->m_leaf = oct->m_valid | (1 << index);
      int cntr = 0;

      tmp = oct->m_child;
      for(int i = 0; i < 8; ++i)
      {
        if((i >> oct->m_valid) & 1)
        {
          cntr++;
        }
      }

      oct->m_child = (unsigned long) new BOct[cntr + 1];

      cntr = 0;
      int j = 0;
      for(int i = 0; i < 8; ++i)
      {
        if((i >> oct->m_valid) & 1)
        {
          // copy pre-existing octants
          (reinterpret_cast<BOct*>(oct->m_child))[cntr++] = (reinterpret_cast<BOct*>(tmp))[j++];
        }

        if(i == index)
        {
          // The new octant should already be default constructed.
          // set position for later
          position = cntr++;
          oct->m_valid = oct->m_valid | (1 << index);
        }
      }
    }
    else
    {
      // find the position
      for(int i = 0; i < position; ++i)
      {
        if((i>>oct->m_valid) & 1)
        {
          position++;
        }
      }
    }
   
    insertPoint(point,
        reinterpret_cast<BOct*> (oct->m_child + position),
        BoundingBox<Vec>(bboxLowerLeft, bboxTopRight)); 
  }

}

