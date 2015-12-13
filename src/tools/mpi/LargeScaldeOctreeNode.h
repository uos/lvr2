//
// Created by eiseck on 08.12.15.
//

#ifndef LAS_VEGAS_LARGESCALDEOCTREENODE_H
#define LAS_VEGAS_LARGESCALDEOCTREENODE_H

#include <lvr/geometry/BoundingBox.hpp>

namespace lvr
{
    class LargeScaldeOctreeNode {
    private:
        float m_size;
        Vertexf m_center;
        LargeScaldeOctreeNode* m_children[8];
        NodeData m_data;
        const size_t m_maxPoints;
    public:
        LargeScaldeOctreeNode(Vertexf center, float size) : m_center(center), m_size(size), m_maxPoints(100000), m_data()
        {
            //cout << "NEW NODE" << endl;
            for(int i=0 ; i<8 ; i++) m_children[i] = NULL;
        }
/*        LargeScaldeOctreeNode(const LargeScaldeOctreeNode& copy) : m_center(copy.m_center), m_size(copy.m_size), m_data(copy.m_data), m_maxPoints(10)
        {

        }*/
        virtual ~LargeScaldeOctreeNode()
        {
            for(int i=0 ; i<8 ; i++)
            {
                if(m_children[i]!=NULL) delete m_children[i];
            }
        }

        bool isLeaf()
        {
            return m_children[0] == NULL;
        }
        void insert(Vertexf pos)
        {
            if(isLeaf())
            {
                m_data.add(pos);
                if(m_data.size() > m_maxPoints)
                {
                    //Todo: splitup octree
                    for(int i = 0 ; i<8 ; i++)
                    {
                        Vertexf newCenter;
                        newCenter.x = m_center.x + m_size * 0.5 * (i&4 ? 0.5 : -0.5);
                        newCenter.y = m_center.y + m_size * 0.5 * (i&2 ? 0.5 : -0.5);
                        newCenter.z = m_center.z + m_size * 0.5 * (i&1 ? 0.5 : -0.5);
                        m_children[i] = new LargeScaldeOctreeNode(newCenter, m_size * 0.5);
                    }
                    for(auto it = m_data.begin() ; it!=m_data.end() ; it++ )
                    {
                        m_children[getOctantContainingPoint(*it)]->insert(*it);
                    }
                    m_data.remove();
                    //m_children[getOctantContainingPoint(pos)]->insert(pos);
                }
            }
            else
            {
                m_children[getOctantContainingPoint(pos)]->insert(pos);
                //Todo: insert recursively
            }

        }

        int getOctantContainingPoint(const Vertexf& point) const
        {
            int oct = 0;
            if(point.x >= m_center.x) oct |= 4;
            if(point.y >= m_center.y) oct |= 2;
            if(point.z >= m_center.z) oct |= 1;
            return oct;
        }


    };
}



#endif //LAS_VEGAS_LARGESCALDEOCTREENODE_H
