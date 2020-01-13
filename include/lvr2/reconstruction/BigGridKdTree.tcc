/**
 * Copyright (c) 2018, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * BigGridKdTree.cpp
 *
 *  Created on: Aug 30, 2017
 *      Author: Isaak Mitschke
 */

#include "lvr2/io/Timestamp.hpp"
#include "lvr2/reconstruction/BigGridKdTree.hpp"

namespace lvr2
{

template <typename BaseVecT>
float BigGridKdTree<BaseVecT>::s_voxelsize;

template <typename BaseVecT>
size_t BigGridKdTree<BaseVecT>::s_maxNodePoints;

template <typename BaseVecT>
BigGrid<BaseVecT>* BigGridKdTree<BaseVecT>::m_grid;

template <typename BaseVecT>
std::vector<BigGridKdTree<BaseVecT>*> BigGridKdTree<BaseVecT>::s_nodes;

template <typename BaseVecT>
BigGridKdTree<BaseVecT>::BigGridKdTree(lvr2::BoundingBox<BaseVecT>& bb,
                                       size_t maxNodePoints,
                                       BigGrid<BaseVecT>* grid,
                                       float voxelsize,
                                       size_t numPoints)
    :

      m_bb(bb), m_numPoints(numPoints)

{
    s_maxNodePoints = maxNodePoints;
    s_voxelsize = voxelsize;
    s_nodes.push_back(this);
    m_grid = grid;
}

template <typename BaseVecT>
BigGridKdTree<BaseVecT>::BigGridKdTree(lvr2::BoundingBox<BaseVecT>& bb, size_t numPoints)
    : m_bb(bb), m_numPoints(0)
{
    s_nodes.push_back(this);
    insert(numPoints, m_bb.getCentroid());
}

template <typename BaseVecT>
BigGridKdTree<BaseVecT>::~BigGridKdTree()
{
    if (this == s_nodes[0] && s_nodes.size() > 0)
    {
        for (int i = 1; i < s_nodes.size(); i++)
        {
            delete s_nodes[i];
        }
        s_nodes.clear();
    }
}

template <typename BaseVecT>
void BigGridKdTree<BaseVecT>::insert(size_t numPoints, BaseVecT pos)
{
    // if kd-tree already exists do this, maybe? this code will never be executed
    if (m_children.size() > 0)
    {
        for (int i = 0; i < m_children.size(); i++)
        {
            if (m_children[i]->fitsInBox(pos))
            {
                m_children[i]->insert(numPoints, pos);
                break;
            }
        }
    }
    else
    {

        // If the new size is larger then max. size, split tree
        if (m_numPoints + numPoints > s_maxNodePoints)
        {

            // Split at X-Axis
            lvr2::BoundingBox<BaseVecT> leftbb;
            lvr2::BoundingBox<BaseVecT> rightbb;
            bool ignoreSplit = false;

            if (m_bb.getXSize() >= m_bb.getYSize() && m_bb.getXSize() >= m_bb.getZSize())
            {
                float left_size = m_bb.getXSize() / 2.0;
                float split_value = m_bb.getMin().x + ceil(left_size / s_voxelsize) * s_voxelsize;

                leftbb = lvr2::BoundingBox<BaseVecT>(
                    BaseVecT(m_bb.getMin().x, m_bb.getMin().y, m_bb.getMin().z),
                    BaseVecT(split_value, m_bb.getMax().y, m_bb.getMax().z));

                rightbb = lvr2::BoundingBox<BaseVecT>(
                    BaseVecT(split_value, m_bb.getMin().y, m_bb.getMin().z),
                    BaseVecT(m_bb.getMax().x, m_bb.getMax().y, m_bb.getMax().z));

                if (leftbb.getXSize() == 0 || rightbb.getXSize() == 0)
                {
                    /*
                    std::cout << leftbb << std::endl;
                    std::cout << rightbb << std::endl;
                    std::cerr
                        << "Error: Requested Maximum Leafsize is Smaller than a points in a
                    voxel(X)"
                        << std::endl;
                    exit(1);
                    */
                    ignoreSplit = true;
                    std::cout << "WARNING: m_numPoints + numPoints = " << m_numPoints + numPoints
                              << " > " << s_maxNodePoints << ". Ignoring x-split" << std::endl;
                }
            }
            // Split at Y-Axis
            else if (m_bb.getYSize() >= m_bb.getXSize() && m_bb.getYSize() >= m_bb.getZSize())
            {

                float left_size = m_bb.getYSize() / 2.0;
                float split_value = m_bb.getMin().y + ceil(left_size / s_voxelsize) * s_voxelsize;

                leftbb = lvr2::BoundingBox<BaseVecT>(
                    BaseVecT(m_bb.getMin().x, m_bb.getMin().y, m_bb.getMin().z),
                    BaseVecT(m_bb.getMax().x, split_value, m_bb.getMax().z));

                rightbb = lvr2::BoundingBox<BaseVecT>(
                    BaseVecT(m_bb.getMin().x, split_value, m_bb.getMin().z),
                    BaseVecT(m_bb.getMax().x, m_bb.getMax().y, m_bb.getMax().z));

                if (leftbb.getYSize() == 0 || rightbb.getYSize() == 0)
                {
                    /*
                    std::cerr
                        << "Error: Requested Maximum Leafsize is Smaller than a points in a
                    voxel(Y)"
                        << std::endl;
                    exit(1);
                    */
                    ignoreSplit = true;
                    std::cout << "WARNING: m_numPoints + numPoints = " << m_numPoints + numPoints
                              << " > " << s_maxNodePoints << ". Ignoring y-split" << std::endl;
                }
            }
            // Split at Z-Axis
            else
            {
                float left_size = m_bb.getZSize() / 2.0;
                float split_value = m_bb.getMin().z + ceil(left_size / s_voxelsize) * s_voxelsize;

                leftbb = lvr2::BoundingBox<BaseVecT>(
                    BaseVecT(m_bb.getMin().x, m_bb.getMin().y, m_bb.getMin().z),
                    BaseVecT(m_bb.getMax().x, m_bb.getMax().y, split_value));

                rightbb = lvr2::BoundingBox<BaseVecT>(
                    BaseVecT(m_bb.getMin().x, m_bb.getMin().y, split_value),
                    BaseVecT(m_bb.getMax().x, m_bb.getMax().y, m_bb.getMax().z));

                if (leftbb.getZSize() == 0 || rightbb.getZSize() == 0)
                {
                    /*
                    std::cerr
                        << "Error: Requested Maximum Leafsize is Smaller than a points in a
                    voxel(Z)"
                        << std::endl;
                    exit(1);
                    */
                    ignoreSplit = true;
                    std::cout << "WARNING: m_numPoints + numPoints = " << m_numPoints + numPoints
                              << " > " << s_maxNodePoints << ". Ignoring z-split" << std::endl;
                }
            }

            if (!ignoreSplit)
            {
                // std::cout << lvr2::timestamp << " rsize start "  << std::endl;
                size_t rightSize = m_grid->getSizeofBox(rightbb.getMin().x,
                                                        rightbb.getMin().y,
                                                        rightbb.getMin().z,
                                                        rightbb.getMax().x,
                                                        rightbb.getMax().y,
                                                        rightbb.getMax().z);
                // std::cout << lvr2::timestamp << " lsize start "  << std::endl;
                size_t leftSize = m_grid->getSizeofBox(leftbb.getMin().x,
                                                       leftbb.getMin().y,
                                                       leftbb.getMin().z,
                                                       leftbb.getMax().x,
                                                       leftbb.getMax().y,
                                                       leftbb.getMax().z);

                // std::cout << lvr2::timestamp << " size_end "  << std::endl;
                BigGridKdTree* leftChild = new BigGridKdTree(leftbb, leftSize);
                BigGridKdTree* rightChild = new BigGridKdTree(rightbb, rightSize);
                m_children.push_back(leftChild);
                m_children.push_back(rightChild);
            }
        }
        else
        {
            m_numPoints += numPoints;
        }
    }
}

template <typename BaseVecT>
std::vector<BigGridKdTree<BaseVecT>*> BigGridKdTree<BaseVecT>::getLeafs()
{
    std::vector<BigGridKdTree*> leafs;
    for (int i = 0; i < s_nodes.size(); i++)
    {
        if (s_nodes[i]->m_children.size() == 0 && s_nodes[i]->m_numPoints > 0)
        {
            leafs.push_back(s_nodes[i]);
        }
    }
    return leafs;
}

} // namespace lvr2
