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
 * NormalAlgorithms.hpp
 *
 * Collection of algorithms for normal calculation.
 *
 * @date 19.07.2017
 * @author Lukas Kalbertodt <lukas.kalbertodt@gmail.com>
 * @author Johan M. von Behren <johan@vonbehren.eu>
 */

#ifndef LVR2_ALGORITHM_NORMALALGORITHMS_H_
#define LVR2_ALGORITHM_NORMALALGORITHMS_H_

#include "lvr2/geometry/BaseMesh.hpp"
#include "lvr2/util/Cluster.hpp"
#include "lvr2/util/ClusterBiMap.hpp"
#include "lvr2/geometry/Normal.hpp"
#include "lvr2/reconstruction/PointsetSurface.hpp"
#include "lvr2/attrmaps/AttrMaps.hpp"

namespace lvr2
{

/**
 * @brief Returns the normal of a face with the given three vertices.
 *
 * @param vertices The face's vertex-positions in counter-clockwise order.
 * @return Either the normal or `none` if the face has a zero area.
 */
template <typename BaseVecT>
boost::optional<Normal<typename BaseVecT::CoordType>> getFaceNormal(array<BaseVecT, 3> vertices);

/**
 * @brief Calculates a normal for each face in the mesh.
 *
 * A face's normal is calculated based on the position of its three vertices.
 * If a face has zero area, the normal cannot be calculated correctly. In this
 * case, a dummy normal (0, 0, 1) is inserted.
 */
template<typename BaseVecT>
DenseFaceMap<Normal<typename BaseVecT::CoordType>> calcFaceNormals(const BaseMesh<BaseVecT>& mesh);

/**
 * @brief Returns a vertex normal for the given vertex interpolated from the
 *        normals of its adjacent faces.
 */
template<typename BaseVecT>
boost::optional<Normal<typename BaseVecT::CoordType>> interpolatedVertexNormal(
    const BaseMesh<BaseVecT>& mesh,
    const FaceMap<Normal<typename BaseVecT::CoordType>>& normals,
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
DenseVertexMap<Normal<typename BaseVecT::CoordType>> calcVertexNormals(
    const BaseMesh<BaseVecT>& mesh,
    const FaceMap<Normal<typename BaseVecT::CoordType>>& normals,
    const PointsetSurface<BaseVecT>& surface
);

/**
 * @brief Calculates a normal for each vertex in the mesh.
 *
 * The normal is calculated by first attempting to interpolate from the
 * adjacent faces. If a vertex doesn't have adjacent faces, the default
 * normal (0, 0, 1) is used.
 *
 * @param surface A point cloud with normal information
 */
template<typename BaseVecT>
DenseVertexMap<Normal<typename BaseVecT::CoordType>> calcVertexNormals(
    const BaseMesh<BaseVecT>& mesh,
    const FaceMap<Normal<typename BaseVecT::CoordType>>& normals
);

} // namespace lvr2

#include "lvr2/algorithm/NormalAlgorithms.tcc"

#endif /* LVR2_ALGORITHM_NORMALALGORITHMS_H_ */
