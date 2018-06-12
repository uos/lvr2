#ifndef POINT_OCTREE
#define POINT_OCTREE

#include <lvr2/io/PointBuffer.hpp>
#include <lvr2/geometry/BaseVector.hpp>
#include <lvr2/geometry/Point.hpp>
#include <lvr2/geometry/BoundingBox.hpp>


namespace lvr2
{

  struct BOct
  {
    public:
      unsigned long m_child : 48;
      unsigned char m_valid : 8;
      unsigned char m_leaf : 8;
  };
  
  template <typename T>
  struct Leaf
  {
    T start;
    T size;
  };

  class PointOctree
  {
    public:
      PointOctree(PointBufferPtr<BaseVector<float> >& points, int voxelSize);

    private:
      int m_voxelSize;
      BOct* m_root;
      
      BoundingBox<BaseVector<float> > m_bbox;
      
      int octant(const Point<BaseVector<float> >& point, const BoundingBox<BaseVector<float> >& bbox, BoundingBox<BaseVector<float> >& sub_bbox);

      inline void insertPoint(const Point<BaseVector<float> >& point, BOct* oct, const BoundingBox<BaseVector<float> >& bbox);

      inline void buildLeaf(const Point<BaseVector<float> >& point, BOct* oct, const )
  };
}

#endif
