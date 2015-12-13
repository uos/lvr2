//
// Created by eiseck on 08.12.15.
//

#ifndef LAS_VEGAS_LargeScaleOctree_H
#define LAS_VEGAS_LargeScaleOctree_H

#include <lvr/geometry/BoundingBox.hpp>
#include "NodeData.hpp"
namespace lvr
{
    class LargeScaleOctree {
    private:
        float m_size;
        Vertexf m_center;
        LargeScaleOctree* m_children[8];
        NodeData m_data;
        const size_t m_maxPoints;
    public:
        LargeScaleOctree(Vertexf center, float size);
        virtual ~LargeScaleOctree();

        bool isLeaf();
        void insert(Vertexf& pos);

        int getOctant(const Vertexf &point) const;


    };
}



#endif //LAS_VEGAS_LargeScaleOctree_H
