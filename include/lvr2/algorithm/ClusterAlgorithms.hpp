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
 * ClusterAlgorithms.hpp
 *
 * Collection of algorithms for cluster modification.
 *
 * @date 24.07.2017
 * @author Johan M. von Behren <johan@vonbehren.eu>
 */

#ifndef LVR2_ALGORITHM_CLUSTERALGORITHMS_H_
#define LVR2_ALGORITHM_CLUSTERALGORITHMS_H_

#include <lvr2/geometry/BaseMesh.hpp>
#include <lvr2/util/ClusterBiMap.hpp>

namespace lvr2
{

template<typename BaseVecT>
void removeDanglingCluster(BaseMesh<BaseVecT>& mesh, size_t sizeThreshold);

/**
 * Finds all contours of an given cluster. An contour can be an "real" boundary, so nothing is adjacent,
 * or an boundary to another edge.
 *
 * @param mesh the mesh to operate on
 * @param clusters map of all clusters
 * @param clusterH the current cluster
 * @return an vector which holds all contours which contains the vertices of the contour.
 */
template<typename BaseVecT>
vector<vector<VertexHandle>> findContours(
    BaseMesh<BaseVecT>& mesh,
    const ClusterBiMap<FaceHandle>& clusters,
    ClusterHandle clusterH
);

/**
 * Simplifies the given contour with the Reumann-Witkam algorithm.
 *
 * @param mesh the mesh to operate on
 * @param contour the contour to be simplified
 * @param threshold angle between edges / lines on one contour
 * @return simplified contour
 */
template<typename BaseVecT>
vector<VertexHandle> simplifyContour(const BaseMesh<BaseVecT>& mesh,
                                     const vector<VertexHandle>& contour,
                                     float threshold
);

} // namespace lvr2

#include <lvr2/algorithm/ClusterAlgorithms.tcc>

#endif /* LVR2_ALGORITHM_CLUSTERALGORITHMS_H_ */
