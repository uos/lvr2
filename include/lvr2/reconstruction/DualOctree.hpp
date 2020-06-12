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
 * DualOctree.hpp
 *
 *  Created on: 18.01.2019
 *      Author: Benedikt Schumacher
 */

#ifndef DualOctree_HPP_
#define DualOctree_HPP_

#include <boost/thread.hpp>
#include "OctreeTables.hpp"

namespace lvr2
{

template<typename BaseVecT, typename BoxT>
class DualLeaf
{
public:

    /**
     * @brief Constructor.
     *
     * @param middle Center of the voxel
     */
    DualLeaf(BaseVecT vertices[]);

    /**
     * @brief Destructor (virtual)
     */
    virtual ~DualLeaf(){};

    /**
     * @brief Calculates and returns the bit-pattern respectively index from the edges of the represented voxel for the MC-Table.
     *
     * @param  distances Distances of the eight edges.
     * @return Index for the MC-Table.
     */
    int getIndex(float distances[]);

    BaseVecT getIntersectionPoint(float intersection, BaseVecT corner_one, BaseVecT corner_two);
    BaseVecT getIntersectionPoint(BaseVecT corner_one, float intersection, BaseVecT corner_two);
    BaseVecT getIntersectionPoint(BaseVecT corner_one, BaseVecT corner_two, float intersection);

    /**
     * @brief Calculates the twelve possible intersections between the cell and the surface to interpolate.
     *
     * @param corners   Eight corners of the current cell.
     * @param distance  Corresponding distance value.
     * @param positions Interpolated intersections.
     */
    void getIntersections(
            BaseVecT corners[],
            float distance[],
            BaseVecT positions[]);

    /**
     * @brief Returns edges of the voxel.
     *
     * @param corner Corners of the voxel.
     */
    void getVertices(
            BaseVecT corners[]);

    /**
     * @brief Returns the stored intersection between the cell and the surface at a given edge.
     *
     * @param i Index of the edge.
     */
    uint getIntersection(char i);

    /**
     * @brief Returns the middle of the represented voxel.
     *
     * @return Middle of the represented voxel.
     */
    BaseVecT& getMiddle();

    /**
     * @brief Sets the intersection between the cell and the surface at a given edge.
     *
     * @param i     Index of the edge.
     * @param value Value of the intersection.
     */
    void setIntersection(char i, uint value);

protected:

    /**
     * @brief Interpolates the intersection between x1 and x1.
     *
     * @param x1 First coordinate.
     * @param x2 Second coordinate.
     * @param d1 Distance value for the first coordinate.
     * @param d2 Distance value for the second coordinate.
     * @return Interpolated distance.
     */
    float calcIntersection(float x1, float x2, float d1, float d2);

    // Vertices of the represented voxel.
    BaseVecT m_vertices[8];

    // Twelve possible intersections.
    uint m_intersections[12];
};

}

#include "DualOctree.tcc"

#endif /* DualOctree_HPP_ */
