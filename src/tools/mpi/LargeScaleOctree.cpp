//
// Created by eiseck on 08.12.15.
//

#include "LargeScaleOctree.hpp"
namespace lvr
{

vector<LargeScaleOctree*> LargeScaleOctree::c_nodeList;


LargeScaleOctree::LargeScaleOctree(Vertexf center, float size, unsigned int maxPoints, size_t bufferSize) : m_center(center), m_size(size), m_maxPoints(maxPoints), m_data(bufferSize), m_leaf(true)
{

    LargeScaleOctree::c_nodeList.push_back(this);
}

LargeScaleOctree::~LargeScaleOctree()
{

}

size_t LargeScaleOctree::getSize()
{
    return m_data.size();
}

Vertexf LargeScaleOctree::getCenter()
{
    return m_center;
}

float LargeScaleOctree::getLength()
{
    return m_size;
}

void LargeScaleOctree::insert(Vertexf& pos)
{
    if(isLeaf())
    {
        m_data.addBuffered(pos);

        if(m_data.size() == m_maxPoints)
        {
            m_data.writeBuffer();
            m_leaf = false;
            //cout << "NEW NODES" << endl;
            for(int i = 0 ; i<8 ; i++)
            {
                Vertexf newCenter;
                newCenter.x = m_center.x + m_size * 0.5 * (i&4 ? 0.5 : -0.5);
                newCenter.y = m_center.y + m_size * 0.5 * (i&2 ? 0.5 : -0.5);
                newCenter.z = m_center.z + m_size * 0.5 * (i&1 ? 0.5 : -0.5);
                m_children.push_back( new LargeScaleOctree(newCenter, m_size * 0.5, m_maxPoints) );
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
    return m_children.size()==0;
}

int LargeScaleOctree::getOctant(const Vertexf& point) const
{
    int oct = 0;
    if(point.x >= m_center.x) oct |= 4;
    if(point.y >= m_center.y) oct |= 2;
    if(point.z >= m_center.z) oct |= 1;
    return oct;
}

string LargeScaleOctree::getFilePath()
{
    return m_data.getDataPath();
}

vector<LargeScaleOctree*>& LargeScaleOctree::getChildren()
{
    return m_children;
}


vector<LargeScaleOctree*> LargeScaleOctree::getNodes()
{
    for(int i = 0; i<c_nodeList.size() ; i++)
    {
        c_nodeList[i]->m_data.writeBuffer();
    }
    return LargeScaleOctree::c_nodeList;
}

bool LargeScaleOctree::operator<(  LargeScaleOctree& rhs )
{
    return getSize() < rhs.getSize();
}





}
