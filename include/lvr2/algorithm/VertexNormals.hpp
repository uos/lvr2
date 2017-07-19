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
 * VertexNormals.hpp
 *
 *  @date 19.07.2017
 *  @author Johan M. von Behren <johan@vonbehren.eu>
 */

#ifndef LVR2_ALGORITHM_VERTEXNORMALS_H_
#define LVR2_ALGORITHM_VERTEXNORMALS_H_

#include <lvr2/geometry/BaseMesh.hpp>
#include <lvr2/geometry/Normal.hpp>
#include <lvr2/util/VectorMap.hpp>
#include <lvr2/reconstruction/PointsetSurface.hpp>

namespace lvr2
{

// ==========================================================================
// Collection of algorithms for normal calculation for vertices in a mesh
// ==========================================================================

/**
 * @brief
 * @tparam BaseVecT
 * @param mesh
 * @param normal
 * @return
 */
template<typename BaseVecT>
VertexMap<Normal<BaseVecT>> calcNormalsForMesh(
    const BaseMesh<BaseVecT>& mesh,
    const PointsetSurface<BaseVecT>& surface
);

} // namespace lvr2

#include <lvr2/algorithm/VertexNormals.tcc>

#endif /* LVR2_ALGORITHM_VERTEXNORMALS_H_ */
