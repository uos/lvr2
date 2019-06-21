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
 * FastBox.h
 *
 *  Created on: 03.03.2011
 *      Author: Thomas Wiemann
 */

#ifndef _LVR2_RECONSTRUCTION_FASTBOX_H_
#define _LVR2_RECONSTRUCTION_FASTBOX_H_

#include "lvr2/reconstruction/MCTable.hpp"
#include "lvr2/reconstruction/FastBoxTables.hpp"

#include "lvr2/geometry/Normal.hpp"

#include "QueryPoint.hpp"

#include <vector>
#include <limits>

using std::vector;
using std::numeric_limits;

namespace lvr2
{

template<typename BoxT>
struct BoxTraits
{
    const static string type;
};

/**
 * @brief A volume representation used by the standard Marching Cubes
 *        implementation.
 */
template<typename BaseVecT>
class FastBox
{
public:

    /**
     * @brief Constructs a new box at the given center point defined
     *        by the used \ref{m_voxelsize}.
     */
    FastBox(BaseVecT center);

    /**
     * @brief Destructor.NormalT
     */
    virtual ~FastBox() {};

    /**
     * @brief Each cell vertex (0 to 7) as associated with a vertex
     *        in the reconstruction grid. This methods assigns the
     *        index \ref{value} to the \ref{index}th cell corner.
     *
     * @param index         One of the eight cell corners.
     * @param value         An index in the reconstruction grid.
     */
    void setVertex(int index, uint value);

    /**
     * @brief Adjacent cells in the grid should use common vertices.
     *        This functions assigns the value of corner[index] to
     *        the corresponding corner of the give neighbor cell.
     *
     * @param index         One of the eight cell corners.
     * @param cell          A neighbor cell.
     */
    void setNeighbor(int index, FastBox<BaseVecT>* cell);

    /**
     * @brief Gets the vertex index of the queried cell corner.
     *
     * @param index         One of the eight cell corners
     * @return              A vertex index.
     */
    uint getVertex(int index);


    FastBox<BaseVecT>*     getNeighbor(int index);

    inline BaseVecT getCenter() { return m_center; }


    /**
     * @brief Performs a local reconstruction according to the standard
     *        Marching Cubes table from Paul Bourke.
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

    virtual void getSurface(
        BaseMesh<BaseVecT>& mesh,
        vector<QueryPoint<BaseVecT>>& query_points,
        uint& globalIndex,
        BoundingBox<BaseVecT>& bb,
        vector<unsigned int>& duplicates,
        float comparePrecision
    );

    /// The voxelsize of the reconstruction grid
    static float             m_voxelsize;

    /// An index value that is used to reference vertices that are not in the grid
    static uint             INVALID_INDEX;

    /// The twelve intersection between box and surface
    OptionalVertexHandle        m_intersections[12];
    bool                        m_extruded;
    bool                        m_duplicate;

     /// The box center
    BaseVecT m_center;

    /// Pointer to all adjacent cells
    FastBox<BaseVecT>*  m_neighbors[27];

protected:


    inline bool compareFloat(double num1, double num2)
    {
        if(fabs(num1 - num2) < std::numeric_limits<double>::epsilon())
            return true;
        else
            return false;
    }

    /**
     * @brief Calculated the index for the MC table
     */
    int  getIndex(vector<QueryPoint<BaseVecT>>& query_points);

    /**
     * @brief Calculated the 12 possible intersections between
     *        the cell and the surface to interpolate
     *
     * @param corners       The eight corners of the current cell
     * @param distance      The corresponding distance value
     * @param positions     The interpolated intersections.
     */
    void getIntersections(BaseVecT* corners, float* distance, BaseVecT* positions);

    /**
     * @brief Calculates the position of the eight cell corners
     *
     * @param corners       The cell corners
     * @param query_points  The query points of the grid
     */
    void getCorners(BaseVecT corners[], vector<QueryPoint<BaseVecT>>& query_points);

    /**
     * @brief Calculates the distance value for the eight cell corners.
     *
     * @param distances     The distance values
     * @param query_points  The query points of the grid
     */
    void getDistances(float distances[], vector<QueryPoint<BaseVecT>>& query_points);

    /***
     * @brief Interpolates the intersection between x1 and x1.
     *
     * @param x1            The first coordinate
     * @param x2            The second coordinate
     * @param d1            The distance value for the first coordinate
     * @param d2            The distance value for the second coordinate
     * @return The interpolated distance.
     */
    float calcIntersection(float x1, float x2, float d1, float d2);

    float distanceToBB(const BaseVecT& v, const BoundingBox<BaseVecT>& bb) const;


    /// The eight box corners
    uint                        m_vertices[8];

    template <typename T> friend class BilinearFastBox;

    typedef FastBox<BaseVecT> BoxType;
};

} // namespace lvr2

#include "lvr2/reconstruction/FastBox.tcc"

#endif /* _LVR2_RECONSTRUCTION_FASTBOX_H_ */
