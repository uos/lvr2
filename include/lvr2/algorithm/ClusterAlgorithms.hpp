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
 * @author Jan Philipp Vogtherr <jvogtherr@uni-osnabrueck.de>
 * @author Kristin Schmidt <krschmidt@uni-osnabrueck.de>
 */

#ifndef LVR2_ALGORITHM_CLUSTERALGORITHMS_H_
#define LVR2_ALGORITHM_CLUSTERALGORITHMS_H_

#include <lvr2/algorithm/Materializer.hpp>
#include <lvr2/geometry/BaseMesh.hpp>
#include <lvr2/geometry/HalfEdgeMesh.hpp>
#include <lvr2/geometry/Handles.hpp>
#include <lvr2/geometry/Plane.hpp>
#include <lvr2/io/MeshBuffer.hpp>
#include <lvr2/util/Cluster.hpp>
#include <lvr2/util/ClusterBiMap.hpp>

#include <vector>

using std::vector;


namespace lvr2
{

/**
 * Removes all faces in connected clusters which are smaller (have less faces) than sizeThreshold
 *
 * @param mesh the mesh to operate on
 * @param sizeThreshold number of faces which a cluster has to contain to not be deleted
 */
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


/**
 * @brief Calculates contour vertices for a given cluster
 *
 * Calculates the contour vertices for a given cluster. To do so, the
 * algorithm will inspect each edge and find edges that only have one adjacent
 * face as part of the given cluster. These edges are contour edges, so each
 * unique vertex of such an edge will be added to the list of results.
 *
 * @param clusterH cluster handle for given cluster
 * @param mesh the mesh
 * @param clusterBiMap map of clusters for given mesh
 */
template<typename BaseVecT>
vector<VertexHandle> calculateClusterContourVertices(
    ClusterHandle clusterH,
    const BaseMesh<BaseVecT>& mesh,
    const ClusterBiMap<FaceHandle>& clusterBiMap
);

/**
 * @brief Calculates bounding rectangle for a given cluster
 *
 * Calculates a bounding rectangle for a given cluster. To do so, first
 * a regression plane for the cluster must be calculated. It is assumed that the
 * cluster is mostly planar for this to work. With the regression plane the
 * algorithm creates an initial bounding rectangle that encloses all of the
 * clusters vertices. Then, iterative improvement steps rotate the bounding
 * rectangle and calculate the predicted number of texels for a bounding box
 * of this size. The rotation with the least amount of texels will be used.
 *
 * @param contour contour of cluster
 * @param mesh mesh
 * @param cluster cluster
 * @param normals normals
 * @param texelSize texelSize
 * @param clusterH cluster handle (TODO: is not used! remove)
 */
template<typename BaseVecT>
BoundingRectangle<BaseVecT> calculateBoundingRectangle(
    const vector<VertexHandle>& contour,
    const BaseMesh<BaseVecT>& mesh,
    const Cluster<FaceHandle>& cluster,
    const FaceMap<Normal<BaseVecT>>& normals,
    float texelSize,
    ClusterHandle clusterH
);

} // namespace lvr2

#include <lvr2/algorithm/ClusterAlgorithms.tcc>

#endif /* LVR2_ALGORITHM_CLUSTERALGORITHMS_H_ */
