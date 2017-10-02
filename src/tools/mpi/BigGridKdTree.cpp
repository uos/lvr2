//
// Created by imitschke on 30.08.17.
//

#include "BigGridKdTree.hpp"
#include <lvr/io/Timestamp.hpp>

float BigGridKdTree::s_voxelsize;
size_t BigGridKdTree::s_maxNodePoints;
BigGrid* BigGridKdTree::m_grid;
std::vector<BigGridKdTree*> BigGridKdTree::s_nodes;
BigGridKdTree::BigGridKdTree(lvr::BoundingBox<lvr::Vertexf>& bb, size_t maxNodePoints, BigGrid* grid, float voxelsize, size_t numPoints) :


        m_bb(bb),
        m_numPoints(numPoints)

{
    s_maxNodePoints=maxNodePoints;
    s_voxelsize=voxelsize;
    s_nodes.push_back(this);
    m_grid = grid;
}
BigGridKdTree::BigGridKdTree(lvr::BoundingBox<lvr::Vertexf>& bb, size_t numPoints) :
        m_bb(bb),
        m_numPoints(0)
{
    s_nodes.push_back(this);
    insert(numPoints, m_bb.getCentroid());
}
BigGridKdTree::~BigGridKdTree()
{
    if(this == s_nodes[0] && s_nodes.size()>0)
    {
        for(int i = 1 ; i<s_nodes.size() ;i++)
        {
            delete s_nodes[i];
        }
        s_nodes.clear();
    }
}

void BigGridKdTree::insert(size_t numPoints, lvr::Vertexf pos)
{


    if(m_children.size()>0)
    {
        for(int i = 0 ; i < m_children.size() ; i++)
        {
            if(m_children[i]->fitsInBox(pos))
            {
                m_children[i]->insert(numPoints, pos);
                break;
            }
        }
    }
    else
    {
        //If the new size is larger then max. size, split tree
        if(m_numPoints + numPoints > s_maxNodePoints)
        {

            // Split at X-Axis
            lvr::BoundingBox<lvr::Vertexf> leftbb;
            lvr::BoundingBox<lvr::Vertexf> rightbb;

            if( m_bb.getXSize() >= m_bb.getYSize() && m_bb.getXSize() >= m_bb.getZSize() )
            {
                float left_size =  m_bb.getXSize() / 2.0;
                float split_value = m_bb.getMin().x + ceil(left_size / s_voxelsize) * s_voxelsize;

                leftbb = lvr::BoundingBox<lvr::Vertexf>(
                        m_bb.getMin().x, m_bb.getMin().y, m_bb.getMin().z,
                        split_value, m_bb.getMax().y, m_bb.getMax().z
                    );

                rightbb = lvr::BoundingBox<lvr::Vertexf>(
                        split_value, m_bb.getMin().y,  m_bb.getMin().z,
                        m_bb.getMax().x, m_bb.getMax().y, m_bb.getMax().z
                    );

                if(leftbb.getXSize() == 0 || rightbb.getXSize() == 0)
                {
                    std::cerr << "Error: Requested Maximum Leafsize is Smaller than a points in a voxel" << std::endl;
                    exit(1);
                }
            }
            // Split at Y-Axis
            else if( m_bb.getYSize() >= m_bb.getXSize() && m_bb.getYSize() >= m_bb.getZSize() )
            {

                float left_size =  m_bb.getYSize() / 2.0;
                float split_value = m_bb.getMin().y + ceil(left_size / s_voxelsize) * s_voxelsize;

                leftbb = lvr::BoundingBox<lvr::Vertexf>(
                        m_bb.getMin().x, m_bb.getMin().y, m_bb.getMin().z,
                        m_bb.getMax().x, split_value, m_bb.getMax().z
                    );

                rightbb = lvr::BoundingBox<lvr::Vertexf>(
                        m_bb.getMin().x, split_value, m_bb.getMin().z,
                        m_bb.getMax().x, m_bb.getMax().y, m_bb.getMax().z
                    );

                if(leftbb.getYSize() == 0 || rightbb.getYSize() == 0)
                {
                    std::cerr << "Error: Requested Maximum Leafsize is Smaller than a points in a voxel" << std::endl;
                    exit(1);
                }
            }
            // Split at Z-Axis
            else
            {
                float left_size =  m_bb.getZSize() / 2.0;
                float split_value = m_bb.getMin().z + ceil(left_size / s_voxelsize) * s_voxelsize;

                leftbb = lvr::BoundingBox<lvr::Vertexf>(
                        m_bb.getMin().x, m_bb.getMin().y, m_bb.getMin().z,
                        m_bb.getMax().x, m_bb.getMax().y, split_value
                    );

                rightbb = lvr::BoundingBox<lvr::Vertexf>(
                        m_bb.getMin().x, m_bb.getMin().y,  split_value,
                        m_bb.getMax().x, m_bb.getMax().y, m_bb.getMax().z
                    );

                if(leftbb.getZSize() == 0 || rightbb.getZSize() == 0)
                {
                    std::cerr << "Error: Requested Maximum Leafsize is Smaller than a points in a voxel" << std::endl;
                    exit(1);
                }
            }



//            std::cout << lvr::timestamp << " rsize start "  << std::endl;
            size_t rightSize = m_grid->getSizeofBox(rightbb.getMin().x, rightbb.getMin().y, rightbb.getMin().z, rightbb.getMax().x, rightbb.getMax().y, rightbb.getMax().z);
//            std::cout << lvr::timestamp << " lsize start "  << std::endl;
            size_t leftSize = m_grid->getSizeofBox(leftbb.getMin().x, leftbb.getMin().y, leftbb.getMin().z, leftbb.getMax().x, leftbb.getMax().y, leftbb.getMax().z);

//            std::cout << lvr::timestamp << " size_end "  << std::endl;
            BigGridKdTree* leftChild = new  BigGridKdTree(leftbb, leftSize);
            BigGridKdTree* rightChild = new  BigGridKdTree(rightbb, rightSize);
            m_children.push_back(leftChild);
            m_children.push_back(rightChild);
        }
        else
        {
            m_numPoints+=numPoints;
        }
    }

}

std::vector<BigGridKdTree*> BigGridKdTree::getLeafs()
{
    std::vector<BigGridKdTree*> leafs;
    for(int i  = 0 ; i<s_nodes.size() ; i++)
    {
        if(s_nodes[i]->m_children.size() == 0 && s_nodes[i]->m_numPoints > 0)
        {
            leafs.push_back(s_nodes[i]);
        }
    }
    return leafs;
}
