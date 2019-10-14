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



#include "lvr2/reconstruction/VirtualGrid.hpp"

#include <lvr2/io/Timestamp.hpp>

namespace lvr2
{

template <typename BaseVecT>
VirtualGrid<BaseVecT>::VirtualGrid(BoundingBox<BaseVecT>& bb,
                                  size_t maxNodePoints,
                                  size_t gridCellSize,
                                  float voxelsize)
{
 m_pcbb = bb;
 m_gridCellSize = gridCellSize;
 m_voxelsize = voxelsize;
}

template <typename BaseVecT>
VirtualGrid<BaseVecT>::~VirtualGrid()
{
    std::cout << "Bye" << std::endl;
}

template <typename BaseVecT>
void VirtualGrid<BaseVecT>::calculateBoxes()
{
    findInitialBox();
    generateNeighbours();
}

template <typename BaseVecT>
void VirtualGrid<BaseVecT>::setBoundingBox(BoundingBox<BaseVecT> bb){
    m_pcbb = bb;
}

template <typename BaseVecT>
void VirtualGrid<BaseVecT>::findInitialBox()
{
    int min_x = (floor(m_pcbb.getMin().x/m_gridCellSize)) * m_gridCellSize;
    int min_y = (floor(m_pcbb.getMin().y/m_gridCellSize)) * m_gridCellSize;
    int min_z = (floor(m_pcbb.getMin().z/m_gridCellSize)) * m_gridCellSize;
    int max_x = min_x + m_gridCellSize;
    int max_y = min_y + m_gridCellSize;
    int max_z = min_z + m_gridCellSize;

    m_initbox = lvr2::BoundingBox<BaseVecT>(BaseVecT(min_x,min_y, min_z), BaseVecT(max_x, max_y, max_z));

    std::cout << "DOES BOX 1 EVEN COLLIDE IF PC? " << m_initbox.overlap(m_pcbb) << std::endl;
    std::cout << m_initbox << std::endl;
    std::cout << m_pcbb << std::endl;

}

template <typename BaseVecT>
void VirtualGrid<BaseVecT>::generateNeighbours()
{
    if(false) //TODO: think of something appropriate
    {
        findInitialBox();
    }

    // Calculates the numbers of Boxes that fits per axis
    int n_xboxes = ceil((m_pcbb.getXSize() + abs(m_pcbb.getMin().x - m_initbox.getMin().x))/m_gridCellSize);
    int n_yboxes = ceil((m_pcbb.getYSize() + abs(m_pcbb.getMin().y - m_initbox.getMin().y))/m_gridCellSize);
    int n_zboxes = ceil((m_pcbb.getZSize() + abs(m_pcbb.getMin().z - m_initbox.getMin().z))/m_gridCellSize);

    lvr2::BoundingBox<BaseVecT> first = m_initbox;

    for(int i = 0; i < n_xboxes ; i++)
    {
        for(int j = 0; j < n_yboxes; j++)
        {
            for(int h = 0; h < n_zboxes; h++)
            {
                std::shared_ptr<lvr2::BoundingBox<BaseVecT>> next = std::shared_ptr<lvr2::BoundingBox<BaseVecT>>(new lvr2::BoundingBox<BaseVecT>(
                                                                               BaseVecT(first.getMin().x + i * m_gridCellSize, first.getMin().y + j * m_gridCellSize, first.getMin().z + h * m_gridCellSize),
                                                                               BaseVecT(first.getMax().x + i * m_gridCellSize, first.getMax().y + j * m_gridCellSize, first.getMax().z + h * m_gridCellSize)));
                std:cout << "generated a Box :3" << std::endl;
                m_boxes.push_back(next);
            }
        }
    }

}


} // namespace lvr2
