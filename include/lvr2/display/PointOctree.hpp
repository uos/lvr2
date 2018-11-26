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

#ifndef POINT_OCTREE
#define POINT_OCTREE

#include <vector>

#include <lvr2/io/PointBuffer.hpp>
#include <lvr2/geometry/BaseVector.hpp>
#include <lvr2/geometry/Vector.hpp>
#include <lvr2/geometry/BoundingBox.hpp>



namespace lvr2
{

  struct BOct
  {
      unsigned long m_child : 48;
      unsigned char m_valid : 8;
      unsigned char m_leaf : 8;
  };
  
  struct Leaf
  {
    int m_start;
    int m_size;
  };

  struct TmpLeaf
  {
    std::vector<Vector<BaseVector<float> > > pts;
  };

  template <typename T>
  struct isLeaf
  {
    static const bool val = false;
  };

  template <>
  struct isLeaf<TmpLeaf>
  {
     static const bool val = true;
  };

  class PointOctree
  {
    public:
      PointOctree(PointBufferPtr& points, int voxelSize);

      virtual ~PointOctree();

    private:
      int m_voxelSize;
      BOct* m_root;
      
      BoundingBox<BaseVector<float> > m_bbox;
      
      int getBBoxIndex(const Vector<BaseVector<float> >& point, const BoundingBox<BaseVector<float> >& bbox, BoundingBox<BaseVector<float> >& subOctBbox);

      void insertPoint(const Vector<BaseVector<float> >& point, BOct* oct, const BoundingBox<BaseVector<float> >& bbox);

      template <typename T>
      int getOctant(BOct* oct, int index);

      /* return is first free index in serial Buffer */
      void serializePointBuffer(BOct* oct, std::vector<Vector<BaseVector<float> > >& pts);
      
      void clear(BOct* oct);

  };

} // namespace lvr2

#endif
