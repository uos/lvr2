/* Copyright (C) 2015 Uni Osnabrück
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
 * FastKinFuBox.h
 *
 *  Created on: 12.12.2015
 *      Author: Tristan Igelbrink
 */

#ifndef FastKinFuBox_H_
#define FastKinFuBox_H_

#include <lvr/geometry/Vertex.hpp>
#include <lvr/geometry/Normal.hpp>
#include <lvr/reconstruction/FastBox.hpp>
#include <lvr/geometry/HalfEdgeKinFuMesh.hpp>
#include <vector>
#include <limits>

using std::vector;
using std::numeric_limits;

namespace lvr
{

/**
 * @brief A lvr-kinfu volume representation used by the standard Marching Cubes
 *           implementation.
 */
template<typename VertexT, typename NormalT>
class FastKinFuBox : public FastBox<VertexT, NormalT>
{
public:

    /**
     * @brief Constructs a new box at the given center point defined
     *           by the used \ref{m_voxelsize}.
     */
    FastKinFuBox(VertexT &center, bool fusionBox = false, bool m_oldfusionBox = false);

    /**
     * @brief Destructor.
     */
    virtual ~FastKinFuBox() {};

    /**
     * @brief mark this box as a mergebox
     *
     * @param index            One of the eight cell corners.
     * @param cell            A neighbor cell.
     */
    void setFusion(bool fusionBox);

    /**
     * @brief Performs a local reconstruction according to the standard
     *           Marching Cubes table from Paul Bourke. Additional merge vertices
     *        are marked in the mesh
     *
     * @param kinfumesh            The reconstructed mesh
     * @param query_points    A vector containing the query points of the
     *                         reconstruction grid
     * @param globalIndex    The index of the newest vertex in the mesh, i.e.
     *                         a newly generated vertex shout have the index
     *                         globalIndex + 1.
     */
    virtual void getSurface(
            BaseMesh<VertexT, NormalT> &mesh,
            vector<QueryPoint<VertexT> > &query_points,
            uint &globalIndex);

    bool                         m_fusionBox;
    bool                         m_fusedBox;
    bool                        m_oldfusionBox;
    bool                        m_fusionNeighborBox;
};

} // namespace lvr

#include "FastKinFuBox.tcc"

#endif /* FastKinFuBox_H_ */