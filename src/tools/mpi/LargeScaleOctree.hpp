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
        static vector<LargeScaleOctree*> c_nodeList;
        float m_size;
        Vertexf m_center;
        LargeScaleOctree* m_children[8];
        NodeData m_data;
        const size_t m_maxPoints;
        vector<LargeScaleOctree*> leafs;
    public:
        LargeScaleOctree(Vertexf center, float size);
        virtual ~LargeScaleOctree();

        bool isLeaf();
        void insert(Vertexf& pos);
        size_t getSize();
        int getOctant(const Vertexf &point) const;
        LargeScaleOctree* getChildren();
        string getFilePath();
        vector<LargeScaleOctree*> getNodes();


    };
}



#endif //LAS_VEGAS_LargeScaleOctree_H
