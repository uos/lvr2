#ifndef CHUNKED_MESH_OCTREE
#define CHUNKED_MESH_OCTREE

#include <vector>

#include "lvr2/display/BOct.hpp"
#include "lvr2/io/PointBuffer.hpp"
#include "lvr2/geometry/BaseVector.hpp"
#include "lvr2/geometry/BoundingBox.hpp"

#include "lvr2/display/MemoryHandler.hpp"

namespace lvr2
{
//
//  struct BOct
//  {
//      long long m_child : 48;
//      unsigned char m_valid : 8;
//      unsigned char m_leaf : 8;
//      BOct(): m_child(0), m_valid(0), m_leaf(0){}
//  };
  
  struct ChunkLeaf
  {
      std::vector<BaseVector<float> > m_centroids;
      std::vector<size_t> m_hashes;
//      unsigned long long m_mesh : 56;
//      unsigned char m_loaded : 8;
//      ChunkLeaf(): m_mesh(0), m_loaded(0){}
  };

  template <typename BaseVecT>
  class MeshOctree
  {
    public:
      MeshOctree(float voxelSize, std::vector<size_t>& hashes, std::vector<BaseVecT>& centroids, BoundingBox<BaseVecT>& bb);

      //void intersect(double planes[6][4], std::vector<unsigned int>& indices);
      void intersect(double planes[24], std::vector<BaseVecT>& indices, std::vector<size_t>& hashes);
      void setLOD(unsigned char lod) { m_lod = lod; }

      void genDisplayLists() { genDisplayLists(m_root); }

      virtual ~MeshOctree() { m_root = NULL; }

    private:
      float m_voxelSize;
      BOct* m_root;
      BoundingBox<BaseVecT> m_bbox;
      // needs [] operator and has to be strict linear in memory
//      FloatChannel m_points;  
    
//      std::vector<size_t> m_hashes;
//      std::vector<BaseVecT> m_centroid;
      

      ChunkMemoryHandler m_mem;
      
      unsigned char m_lod;
      size_t numLeafs;

      template <typename T>
      void link(BOct* parent, T* child);

      template <typename T>
        T* getChildPtr(BOct* parent);
      
      unsigned char getIndex(const BaseVecT& point, const BoundingBox<BaseVecT>& bbox);

      void getBBoxes(const BoundingBox<BaseVecT>& bbox, BoundingBox<BaseVecT>* boxes);
      
//      template <typename PtrT>
//      void sortPC(size_t start, size_t size, const BoundingBox<BaseVecT>& bbox, size_t bucket_sizes[8]);

      long buildTree(BOct* oct, std::vector<size_t>& hashes, std::vector<BaseVecT>& centroids, const BoundingBox<BaseVecT>& bbox);

      
      void intersect(ChunkLeaf* leaf, const BoundingBox<BaseVecT>& bbox, double planes[24], std::vector<BaseVecT >& indices, std::vector<size_t>& hashes);

      void intersect(BOct* oct, const BoundingBox<BaseVecT>& bbox, double planes[24], std::vector<BaseVecT >& indices, std::vector<size_t>& hashes);
  

    void getHashes(BOct* oct, std::vector<BaseVecT>& indices, std::vector<size_t>& hashes);

//      void colorAndWrite(BOct* oct);
//
//      void colorAndWrite(BOct* oct, unsigned char index);
//      
//      void writeChunkLeaf(Leaf* leaf, unsigned char index);

    //  void intersect(BOct* oct,  const BoundingBox<BaseVecT>& octBBox, const BoundingBox<BaseVecT>& cullBBox, std::vector<BaseVecT >& pts);

    //  void intersect(const BoundingBox<BaseVecT>& cullBBox, std::vector<BaseVecT >& pts);

  };
}

#include "lvr2/display/MeshOctree.tcc"

#endif
