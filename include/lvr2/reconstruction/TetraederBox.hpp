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

/**
 * TetraederBox.h
 *
 *  @date 23.11.2011
 *  @author Thomas Wiemann
 */

#ifndef TETRAEDERBOX_H_
#define TETRAEDERBOX_H_


#include "FastBox.hpp"

namespace lvr2
{

/**
 * @brief Used for Marching Tetreader Reconstruction. Dives a regular box
 *        into 5 Tetraeders for mesh generation.
 */
template<typename BaseVecT>
class TetraederBox : public FastBox<BaseVecT>
{
public:

    /// Creates a new tetraeder box with current grid voxel size
    /// around the given center point
    TetraederBox(BaseVecT center);
    virtual ~TetraederBox();

    /**
     * @brief Performs a local reconstruction using a tetraeder decomposition
     *        of the current cell
     *
     * @param mesh          The reconstructed mesh
     * @param query_points  A vector containing the query points of the
     *                      reconstruction grid
     * @param globalIndex   The index of the newest vertex in the mesh, i.e.
     *                      a newly generated vertex shout have the index
     *                      globalIndex + 1.
     */
    virtual void getSurface(
        BaseMesh<BaseVecT>& mesh,
        vector<QueryPoint<BaseVecT>>& query_points,
        uint &globalIndex
    );

//    virtual void getSurface(
//        BaseMesh<BaseVecT>& mesh,
//        vector<QueryPoint<BaseVecT>>& query_points,
//        uint& globalIndex,
//        BoundingBox<BaseVecT>& bb,
//        vector<unsigned int>& duplicates,
//        float comparePrecision
//    );

private:

    int calcPatternIndex(float distances[4])
    {
        int index = 0;
        for(int i = 0; i < 4; i++)
        {
            if(distances[i] > 0) index |= (1 << i);
        }
        return index;
    }

    inline void interpolateIntersections(
            int tetraNumber,
            BaseVecT positions[4],
            float distances[4]
            );

    OptionalVertexHandle    m_intersections[19];
    BaseVecT         m_intersectionPositionsTetraeder[6];

};

} /* namespace lvr */

#include "TetraederBox.tcc"

#endif /* TETRAEDERBOX_H_ */
