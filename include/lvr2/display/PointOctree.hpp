#ifndef POINT_OCTREE
#define POINT_OCTREE

#include <vector>

#include "lvr2/io/PointBuffer.hpp"
#include "lvr2/display/BOct.hpp"
#include "lvr2/geometry/BaseVector.hpp"
#include "lvr2/geometry/BoundingBox.hpp"

#include "lvr2/display/MemoryHandler.hpp"

namespace lvr2
{

//  struct BOct
//  {
//      long m_child : 48;
//      unsigned char m_valid : 8;
//      unsigned char m_leaf : 8;
//      BOct(): m_child(0), m_valid(0), m_leaf(0){}
//  };
  
  struct Leaf
  {
    unsigned int m_start;
    unsigned int m_size;
    unsigned int m_listIndex;
  };

  template <typename BaseVecT>
  class PointOctree
  {
    public:
      PointOctree(PointBufferPtr& points, int depth);

      void intersect(double planes[24], std::vector<unsigned int>& indices);
      void setLOD(unsigned char lod) { m_lod = lod; }

      void genDisplayLists() { genDisplayLists(m_root); }

      virtual ~PointOctree() { m_root = NULL; }

    private:
      float m_voxelSize;
      BOct* m_root;
      BoundingBox<BaseVecT> m_bbox;
      // needs [] operator and has to be strict linear in memory
      FloatChannel m_points;

      ChunkMemoryHandler m_mem;
      
      unsigned char m_lod;

      template <typename T>
      void link(BOct* parent, T* child);

      template <typename T>
        T* getChildPtr(BOct* parent);
      
      unsigned char getIndex(const BaseVecT& point, const BoundingBox<BaseVecT>& bbox);

      void getBBoxes(const BoundingBox<BaseVecT>& bbox, BoundingBox<BaseVecT>* boxes);
      
      template <typename PtrT>
      void sortPC(size_t start, size_t size, const BoundingBox<BaseVecT>& bbox, size_t bucket_sizes[8]);

      long buildTree(BOct* oct, size_t start, size_t size, const BoundingBox<BaseVecT>& bbox);

      void getPoints(BOct* oct, std::vector<unsigned int >& indices);
      
      void intersect(Leaf* leaf, const BoundingBox<BaseVecT>& bbox, double planes[24], std::vector<unsigned int>& indices);

      void intersect(BOct* oct, const BoundingBox<BaseVecT>& bbox, double planes[24], std::vector<unsigned int>& indices);
  
      void genDisplayLists(Leaf* leaf);

      void genDisplayLists(BOct* oct);
 


//      void colorAndWrite(BOct* oct);
//
//      void colorAndWrite(BOct* oct, unsigned char index);
//      
//      void writeLeaf(Leaf* leaf, unsigned char index);

    //  void intersect(BOct* oct,  const BoundingBox<BaseVecT>& octBBox, const BoundingBox<BaseVecT>& cullBBox, std::vector<BaseVecT >& pts);

    //  void intersect(const BoundingBox<BaseVecT>& cullBBox, std::vector<BaseVecT >& pts);


  };
}

#include "lvr2/display/PointOctree.tcc"

#endif
