//
// Created by imitschke on 30.08.17.
//

#ifndef LAS_VEGAS_BIGGRIDKDTREE_H
#define LAS_VEGAS_BIGGRIDKDTREE_H
#include <lvr/geometry/BoundingBox.hpp>
#include <lvr/geometry/Vertex.hpp>
#include "BigGrid.hpp"

#include <memory>
#include <vector>
class BigGridKdTree
{
public:
    BigGridKdTree(lvr::BoundingBox<lvr::Vertexf>& bb, size_t maxNodePoints, BigGrid* grid, float voxelsize, size_t numPoints = 0);
    virtual ~BigGridKdTree();
    void insert(size_t numPoints, lvr::Vertexf pos);
    static std::vector<BigGridKdTree*> getLeafs();
    static std::vector<BigGridKdTree*> getNodes(){return s_nodes;}
    inline size_t getNumPoints(){return m_numPoints;}
    inline lvr::BoundingBox<lvr::Vertexf>&  getBB(){return m_bb;}
private:
    lvr::BoundingBox<lvr::Vertexf> m_bb;
    size_t m_numPoints;
    std::vector<BigGridKdTree*> m_children;
    BigGridKdTree(lvr::BoundingBox<lvr::Vertexf>& bb, size_t numPoints = 0);

    static std::vector<BigGridKdTree*> s_nodes;
    static float s_voxelsize;
    static size_t s_maxNodePoints;
    static BigGrid* m_grid;
    inline bool fitsInBox(lvr::Vertexf& pos) {return m_bb.contains(pos.x, pos.y, pos.z);}

};


#endif //LAS_VEGAS_BIGGRIDKDTREE_H
