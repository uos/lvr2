/* Copyright (C) 2011 Uni Osnabrück
 * This file is part of the LAS VEGAS Reconstruction Toolkit,
 *
 * LAS VEGAS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * LAS VEGAS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA
 */


 /*
 * FastBox.h
 *
 *  Created on: 03.03.2011
 *      Author: Thomas Wiemann
 */

#ifndef FastBox_H_
#define FastBox_H_

#include "geometry/Vertex.hpp"
#include "geometry/Normal.hpp"
#include "reconstruction/QueryPoint.hpp"
#include "reconstruction/MCTable.hpp"
#include "reconstruction/FastBoxTables.hpp"
#include <vector>
#include <limits>

using std::vector;
using std::numeric_limits;

namespace lvr
{

template<typename BoxT>
struct BoxTraits
{
	const static string type;
};

/**
 * @brief A volume representation used by the standard Marching Cubes
 * 		  implementation.
 */
template<typename VertexT, typename NormalT>
class FastBox
{
public:

	/**
	 * @brief Constructs a new box at the given center point defined
	 * 		  by the used \ref{m_voxelsize}.
	 */
    FastBox(VertexT &center);

    /**
     * @brief Destructor.
     */
    virtual ~FastBox() {};

    /**
     * @brief Each cell vertex (0 to 7) as associated with a vertex
     *        in the reconstruction grid. This methods assigns the
     *        index \ref{value} to the \ref{index}th cell corner.
     *
     * @param index			One of the eight cell corners.
     * @param value			An index in the reconstruction grid.
     */
    void setVertex(int index,  uint value);

    /**
     * @brief Adjacent cells in the grid should use common vertices.
     * 		  This functions assigns the value of corner[index] to
     * 		  the corresponding corner of the give neighbor cell.
     *
     * @param index			One of the eight cell corners.
     * @param cell			A neighbor cell.
     */
    void setNeighbor(int index, FastBox<VertexT, NormalT>* cell);

    /**
     * @brief Gets the vertex index of the queried cell corner.
     *
     * @param index			One of the eight cell corners
     * @return				A vertex index.
     */
    uint getVertex(int index);


    FastBox<VertexT, NormalT>*     getNeighbor(int index);

    /**
     * @brief Performs a local reconstruction according to the standard
     * 		  Marching Cubes table from Paul Bourke.
     *
     * @param mesh			The reconstructed mesh
     * @param query_points	A vector containing the query points of the
     * 						reconstruction grid
     * @param globalIndex	The index of the newest vertex in the mesh, i.e.
     * 						a newly generated vertex shout have the index
     * 						globalIndex + 1.
     */
    virtual void getSurface(
            BaseMesh<VertexT, NormalT> &mesh,
            vector<QueryPoint<VertexT> > &query_points,
            uint &globalIndex);

    /// The voxelsize of the reconstruction grid
    static float             m_voxelsize;

    /// An index value that is used to reference vertices that are not in the grid
    static uint		   		INVALID_INDEX;

    /// The twelve intersection between box and surface
    uint                   		m_intersections[12];

protected:

    /**
     * @brief Calculated the index for the MC table
     */
    int  getIndex(vector<QueryPoint<VertexT> > &query_points);

    /**
     * @brief Calculated the 12 possible intersections between
     *        the cell and the surface to interpolate
     *
     * @param corners       The eight corners of the current cell
     * @param distance      The corresponding distance value
     * @param positions     The interpolated intersections.
     */
    void getIntersections(VertexT* corners, float* distance, VertexT* positions);

    /**
     * @brief Calculates the position of the eight cell corners
     *
     * @param corners       The cell corners
     * @param query_points  The query points of the grid
     */
    void getCorners(VertexT corners[], vector<QueryPoint<VertexT> > &query_points);

    /**
     * @brief Calculates the distance value for the eight cell corners.
     *
     * @param distances     The distance values
     * @param query_points  The query points of the grid
     */
    void getDistances(float distances[], vector<QueryPoint<VertexT> > &query_points);



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

    /// The box center
    VertexT               		m_center;

    /// The eight box corners
    uint                  		m_vertices[8];

    /// Pointer to all adjacent cells
    FastBox<VertexT, NormalT>*  m_neighbors[27];

    template<typename Q, typename V> friend class BilinearFastBox;

    typedef FastBox<VertexT, NormalT> BoxType;
};

} // namespace lvr

#include "FastBox.tcc"

#endif /* FastBox_H_ */
