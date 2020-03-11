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
 * VirtualGrid.hpp
 *
 *  @date 17.09.19
 *  @author Pao
 */

#ifndef VIRTUALGRID_H
#define VIRTUALGRID_H
#include "BigGrid.hpp"
#include "lvr2/geometry/BoundingBox.hpp"

#include <memory>
#include <vector>

namespace lvr2
{

template <typename BaseVecT>
class VirtualGrid
{
  public:
    /**
     * Constructor
     *
     * @param bb
     * @param maxNodePoints
     * @param gridCellSize
     * @param voxelsize
     */
    VirtualGrid(BoundingBox<BaseVecT>& bb,
                  size_t gridCellSize,
                  float voxelsize);


    /**
     * Destructor
     */
    virtual ~VirtualGrid();

    /**
     * Method to calculate the boxes/chunks covering the PointCloud-BB
     *
     */
    void calculateBoxes();

    /**
     * Sets a new Bounding Box
     * @param bb
     */
    void setBoundingBox(BoundingBox<BaseVecT> bb);

    /**
     *
     * @return the chunks of the virtual Grid
     */
    std::shared_ptr<std::vector<BoundingBox<BaseVecT>>> getBoxes() { return std::make_shared<std::vector<BoundingBox<BaseVecT>>>(m_boxes); }

  private:
    /**
     * locates the initial Box surrounding the lower left corner of the PointCloud-BB
     *
     */
    void findInitialBox();



    /**
     * generates the Boxes/Chunks surrounding the initial Box to cover the PointCloud-BB
     *
     */
    void generateNeighbours();

    // BoundingBox of the input PointCloud
    BoundingBox<BaseVecT> m_pcbb;

    // initial Bounding Box, around the left corner of the PointCloud-BB
    BoundingBox<BaseVecT> m_initbox;

    // List of (smaller) BoundingBox (aka Chunks), which overlap the original PointCloud
    std::vector<BoundingBox<BaseVecT>> m_boxes;

    // size of the "virtual" GridCell aka ChunkSize
    size_t m_gridCellSize;

    // voxelsize to verify the cellsize/chunksize
    float m_voxelsize;

};

} // namespace lvr2

#include "lvr2/reconstruction/VirtualGrid.tcc"

#endif // VIRTUALGRID_H
