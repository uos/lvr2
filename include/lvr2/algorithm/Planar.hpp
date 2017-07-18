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
 * Planar.hpp
 *
 *  @date 14.07.2017
 *  @author Johan M. von Behren <johan@vonbehren.eu>
 */

#ifndef LVR2_ALGORITHM_PLANAR_H_
#define LVR2_ALGORITHM_PLANAR_H_

#include <lvr2/util/VectorMap.hpp>
#include <lvr2/geometry/ClusterSet.hpp>
#include <lvr2/geometry/Handles.hpp>
#include <lvr2/geometry/BaseMesh.hpp>
#include <lvr2/geometry/Plane.hpp>

namespace lvr2
{

// ==========================================================================
// Collection of algorithms for planar cluster growing.
// ==========================================================================

/**
 * @brief Algorithm which generates plane clusters from the given mesh.
 * @param minSinAngle `1 - minSinAngle` is the allowed difference between the sin of the angle of the starting
 *                    face and all other faces in one cluster.
 */
template<typename BaseVecT>
ClusterSet<FaceHandle> planarClusterGrowing(const BaseMesh<BaseVecT>& mesh, float minSinAngle);

/**
 * @brief Algorithm which generates plane clusters from the given mesh, drags points in clusters into regression planes
 *        and improves clusters iteratively.
 * @param mesh
 * @param minSinAngle `1 - minSinAngle` is the allowed difference between the sin of the angle of the starting
 *                    face and all other faces in one cluster.
 * @param numIterations for cluster improvement
 */
template<typename BaseVecT>
ClusterSet<FaceHandle>
    iterativePlanarClusterGrowing(
        BaseMesh<BaseVecT>& mesh,
        float minSinAngle,
        int numIterations,
        int minRegionSize
    );

template<typename BaseVecT>
Plane<BaseVecT>
    calcRegressionPlane(
        const BaseMesh<BaseVecT>& mesh,
        const Cluster<FaceHandle>& cluster
    );

template<typename BaseVecT>
ClusterMap<Plane<BaseVecT>>
    calcRegressionPlanes(
        const BaseMesh<BaseVecT>& mesh,
        const ClusterSet<FaceHandle>& clusterSet,
        int minRegionSize
    );

template<typename BaseVecT>
void dragToRegressionPlane(
    BaseMesh<BaseVecT>& mesh,
    const Cluster<FaceHandle>& cluster,
    const Plane<BaseVecT>& plane
);

template<typename BaseVecT>
void dragToRegressionPlanes(
    BaseMesh<BaseVecT>& mesh,
    const ClusterSet<FaceHandle>& clusterSet,
    const ClusterMap<Plane<BaseVecT>>& clusterMap
);

template<typename BaseVecT>
void debugPlanes(
    const BaseMesh<BaseVecT>& mesh,
    const ClusterSet<FaceHandle>& clusterSet,
    const ClusterMap<Plane<BaseVecT>>& clusterMap,
    string filename,
    size_t minClusterSize
);

} // namespace lvr2

#include <lvr2/algorithm/Planar.tcc>

#endif /* LVR2_ALGORITHM_PLANAR_H_ */
