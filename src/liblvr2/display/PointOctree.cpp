/**
 * Copyright (c) 2018, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <lvr2/display/PointOctree.hpp>
#include <vector>
#include <algorithm>
#include <cmath>

namespace lvr2
{
  using Vec = BaseVector<float>;

  PointOctree::PointOctree(PointBufferPtr& points, int voxelSize)
  {
    m_voxelSize = voxelSize;

    FloatChannelOptional pts_channel = points->getFloatChannel("points");
    FloatChannel pts_data = *pts_channel;

    // initializ min max for bounding box
    float minX = pts_data[0][0];
    float minY = pts_data[0][1];
    float minZ = pts_data[0][2];
    float maxX = pts_data[0][0];
    float maxY = pts_data[0][1];
    float maxZ = pts_data[0][2];

    //    BoundingBox<Vec> bb;
    for(int i = 0; i < points->numPoints(); ++i)
    {
      Vec p = pts_data[i];
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

    Vec v1(min, min, min);
    Vec v2(max, max, max);

    m_bbox = BoundingBox<Vec>(v1, v2);

    // number of splits to get a resolution smaller than voxelSize
    int depth = std::ceil(std::log2(m_bbox.getLongestSide()/m_voxelSize));

    m_root = new BOct();

    for(int i = 0; i < points->numPoints(); ++i)
    {
      insertPoint(pts_data[i], m_root, m_bbox);
    }

    std::vector<Vec > pts;
    pts.reserve(points->numPoints());

    serializePointBuffer(m_root, pts);

  }

  int PointOctree::getBBoxIndex(const Vec& point, const BoundingBox<Vec >& bbox, BoundingBox<Vec >& subOctBbox)
  {
    int index = 0;
    Vec centroid = bbox.getCentroid();

    Vec bboxLowerLeft = centroid;
    Vec bboxTopRight = centroid;

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
    subOctBbox = BoundingBox<Vec>(bboxLowerLeft, bboxTopRight);
    return index;
  }

  template <typename T>
    int PointOctree::getOctant(BOct* oct, int index)
    {
      int position = 0;
      unsigned char mask = 0;
      if(isLeaf<T>::val)
      {
        mask = oct->m_leaf;
      }
      else
      {
        mask = oct->m_valid;
      }

      // create new oct->m_childant n
      if(!((index >> mask) & 1))
      {
        unsigned long tmp = oct->m_child;
        int cntr = 0;

        tmp = oct->m_child;
        for(int i = 0; i < 8; ++i)
        {
          if((i >> mask) & 1)
          {
            cntr++;
          }
        }

        oct->m_child = (unsigned long) new T[cntr + 1];

        cntr = 0;
        int j = 0;
        for(int i = 0; i < 8; ++i)
        {
          if((i >> mask) & 1)
          {
            // copy pre-existing oct->m_childants
            (reinterpret_cast<T*>(oct->m_child))[cntr++] = (reinterpret_cast<T*>(tmp))[j++];
          }

          if(i == index)
          {
            // The new octant should already be default constructed.
            // set position for later
            position = cntr++;
            mask = mask | (1 << index);
            oct->m_valid = 0;
          }
        }

        if(isLeaf<T>::val)
        {
          oct->m_leaf = mask;
        }
        else
        {
          oct->m_valid = mask;
        } 
      }
      else
      {
        // find the position
        for(int i = 0; i < (index + 1); ++i)
        {
          if((i >> mask) & 1)
          {
            position++;
          }
        }
      }

      return position;
    }

  inline void PointOctree::insertPoint(const Vec& point, BOct* oct, const BoundingBox<Vec >& bbox)
  {
    BoundingBox<Vec > subOctBbox;
    int index = getBBoxIndex(point, bbox, subOctBbox);


    int position = 0;
    // next is leaf
    // bbox is square so no need for comparison.
    if(subOctBbox.getXSize() <= m_voxelSize)
    {
      position = getOctant<TmpLeaf>(oct, index);

      // adjust leaf bitmask as well


      (reinterpret_cast<TmpLeaf* >(oct->m_child))[position].pts.push_back(point);
      oct->m_leaf = oct->m_leaf | (1 << index);
      // recursion anchor
      // now we have build the tree structure without the leaves
      return;
    }

    position = getOctant<BOct>(oct, index);



    insertPoint(point,
        reinterpret_cast<BOct*> (oct->m_child + position),
        subOctBbox
        ); 
  }

  void PointOctree::serializePointBuffer(BOct* oct, std::vector< Vec >& pts)
  {
    if(!oct->m_valid)
    {
      if(!oct->m_leaf)
      {
        // empty oct
        return;
      }

      ssize_t cntr = 0;
      //
      for(int i = 0; i < 8; ++i)
      {
        if((i >> oct->m_leaf) & 1)
        {
          cntr++;
        }
      }

      Leaf* newLeaves = new Leaf[cntr];

      for(int i = 0; i < cntr; ++i)
      {
        TmpLeaf* old = (reinterpret_cast<TmpLeaf*>(oct->m_child)) + i;
        auto it = pts.end();
        newLeaves[i].m_start = it - pts.begin();
        newLeaves[i].m_size = old->pts.size();
        pts.insert(it, old->pts.begin(), old->pts.end());
      }

      // delete temporary leaves.
      delete [] (reinterpret_cast<TmpLeaf*>(oct->m_child));

      // set the more efficient leaves
      oct->m_child = (unsigned long) newLeaves;

      return;
    }

    ssize_t size = 0;
    for(int i = 0; i < 8; ++i)
    {
      if((i >> oct->m_valid) & 1)
      {
        serializePointBuffer((reinterpret_cast<BOct*>(oct->m_child)) + i, pts);
        size++;
      } 
    }

    return;

  }

  void PointOctree::clear(BOct* oct)
  {
    ssize_t cntr;

    if(oct->m_leaf)
    {
      delete [] (reinterpret_cast<Leaf*>(oct->m_child));
      oct->m_leaf = 0;
      return;
    }

    for(int i = 0; i < 8; ++i)
    {
      if((i >> oct->m_valid) & 1)
      {
        cntr++;
      }
    }

    for(int i = 0; i < cntr; ++i)
    {
      clear(reinterpret_cast<BOct*>(oct->m_child) + i);
    }

    delete [] reinterpret_cast<BOct*>(oct->m_child);

    return;
  }

  PointOctree::~PointOctree()
  {
    clear(m_root);
    delete m_root;

  }
} // namespace lvr2
