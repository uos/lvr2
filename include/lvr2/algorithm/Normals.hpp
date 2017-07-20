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
 * Normals.hpp
 *
 * Collection of algorithms for normal calculation.
 *
 *  @date 17.06.2017
 */

#ifndef LVR2_ALGORITHM_NORMALS_H_
#define LVR2_ALGORITHM_NORMALS_H_

#include <lvr2/geometry/BaseMesh.hpp>
#include <lvr2/geometry/Cluster.hpp>
#include <lvr2/geometry/ClusterSet.hpp>
#include <lvr2/geometry/Normal.hpp>
#include <lvr2/reconstruction/PointsetSurface.hpp>
#include <lvr2/util/VectorMap.hpp>

namespace lvr2
{

template <typename BaseVecT>
FaceMap<Normal<BaseVecT>> calcFaceNormals(const BaseMesh<BaseVecT>& mesh);

template <typename BaseVecT>
optional<Normal<BaseVecT>> interpolatedVertexNormal(
    const BaseMesh<BaseVecT>& mesh,
    const FaceMap<Normal<BaseVecT>>& normals,
    VertexHandle handle
);

/**
 * @brief Calculates a normal for each vertex in the mesh.
 *
 * The normal is calculated by first attempting to interpolate from the
 * adjacent faces. If a vertex doesn't have adjacent faces, the normal from
 * the nearest point in the point cloud is used.
 *
 * @param surface A point cloud with normal information
 */
template<typename BaseVecT>
VertexMap<Normal<BaseVecT>> calcVertexNormals(
    const BaseMesh<BaseVecT>& mesh,
    const FaceMap<Normal<BaseVecT>>& normals,
    const PointsetSurface<BaseVecT>& surface
);

} // namespace lvr2

#include <lvr2/algorithm/Normals.tcc>

#endif /* LVR2_ALGORITHM_NORMALS_H_ */
