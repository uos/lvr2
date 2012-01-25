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
 * Reconstructor.h
 *
 *  Created on: 16.02.2011
 *      Author: Thomas Wiemann
 */

#ifndef MESH_GENERATOR_HPP
#define MESH_GENERATOR_HPP

#include "../geometry/BaseMesh.hpp"
#include "../reconstruction/PointsetSurface.hpp"

namespace lssr
{

/**
 * @brief Interface class for surface reconstruction algorithms
 *        that generate triangle meshes from point set surfaces.
 */
template<typename VertexT, typename NormalT>
class MeshGenerator
{
public:

    /**
     * @brief Constructs a Reconstructor object using the given point
     *        cloud handler
     */
    MeshGenerator( typename PointsetSurface<VertexT>::Ptr surface) : m_manager(surface) {}

    /**
     * @brief Generates a triangle mesh representation of the current
     *        point set.
     *
     * @param mesh      A surface representation of the current point
     *                  set.
     */
    virtual void getMesh(BaseMesh<VertexT, NormalT>& mesh) = 0;

protected:

    /// The point cloud manager that handles the loaded point cloud data.
    typename PointsetSurface<VertexT>::Ptr      m_manager;
};

} //namespace lssr

#endif /* MESH_GENERATOR_HPP */
