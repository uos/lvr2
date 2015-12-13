//
// Created by eiseck on 08.12.15.
//

#include "LargeScaleOctree.hpp"
namespace lvr
{
LargeScaleOctree::LargeScaleOctree(Vertexf center, float size) : m_center(center), m_size(size), m_maxPoints(4700000), m_data()
{
    for(int i=0 ; i<8 ; i++) m_children[i] = NULL;
}

LargeScaleOctree::~LargeScaleOctree()
{
    for(int i=0 ; i<8 ; i++)
    {
        if(m_children[i]!=NULL) delete m_children[i];
    }
}
void LargeScaleOctree::insert(Vertexf& pos)
{
    if(isLeaf())
    {
        m_data.add(pos);
        if(m_data.size() == m_maxPoints)
        {
            for(int i = 0 ; i<8 ; i++)
            {
                Vertexf newCenter;
                newCenter.x = m_center.x + m_size * 0.5 * (i&4 ? 0.5 : -0.5);
                newCenter.y = m_center.y + m_size * 0.5 * (i&2 ? 0.5 : -0.5);
                newCenter.z = m_center.z + m_size * 0.5 * (i&1 ? 0.5 : -0.5);
                m_children[i] = new LargeScaleOctree(newCenter, m_size * 0.5);
            }
            for(Vertexf v : m_data)
            {
                m_children[getOctant(v)]->insert(v);
            }
            m_data.remove();

        }

    }
    else
    {
        m_children[getOctant(pos)]->insert(pos);
    }

}

bool LargeScaleOctree::isLeaf()
{
    return m_children[0] == NULL;
}

int LargeScaleOctree::getOctant(const Vertexf& point) const
{
    int oct = 0;
    if(point.x >= m_center.x) oct |= 4;
    if(point.y >= m_center.y) oct |= 2;
    if(point.z >= m_center.z) oct |= 1;
    return oct;
}

}