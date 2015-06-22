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
 * LocalApproximation.hpp
 *
 *  Created on: 15.02.2011
 *      Author: Thomas Wiemann
 */

#ifndef LOCALAPPROXIMATION_HPP_
#define LOCALAPPROXIMATION_HPP_

#include "geometry/BaseMesh.hpp"
#include "reconstruction/AdaptiveKSearchSurface.hpp"

namespace lvr
{

/**
 * @brief	An interface class for local approximation operations
 * 			(e.g. in a Marching Cubes box)
 */
template<typename VertexT, typename NormalT>
class LocalApproximation
{
public:

	/**
	 * @brief	Adds the local reconstruction to the given mesh
	 *
	 * @param	mesh		 The used mesh.
	 * @param	manager		 A point cloud manager object
	 * @param	globalIndex	 The index of the latest vertex in the mesh
	 */
    virtual void getSurface(BaseMesh<VertexT, NormalT> &mesh,
            AdaptiveKSearchSurface<VertexT, NormalT> &manager,
            uint &globalIndex);

};


} // namspace lvr

#endif /* LOCALAPPROXIMATION_HPP_ */
