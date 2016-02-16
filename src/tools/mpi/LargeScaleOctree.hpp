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

        /*
                id:	    0 1 2 3 4 5 6 7
                x:      - - - - + + + +
                y:      - - + + - - + +
                z:      - + - + - + - +
         */
        vector<LargeScaleOctree*>  m_children;
        NodeData m_data;
        const size_t m_maxPoints;
        vector<LargeScaleOctree*> leafs;
        bool m_leaf;
    public:
        LargeScaleOctree(Vertexf center, float size, unsigned int maxPoints, size_t bufferSize = 40000000);
	virtual ~LargeScaleOctree();

        bool operator<(  LargeScaleOctree& rhs ) ;
        bool isLeaf();
        void insert(Vertexf& pos);
        size_t getSize();
        Vertexf getCenter();
        float getLength();
        int getOctant(const Vertexf &point) const;
        vector<LargeScaleOctree*>& getChildren();
        string getFilePath();
        vector<LargeScaleOctree*> getNodes();


    };
}



#endif //LAS_VEGAS_LargeScaleOctree_H
