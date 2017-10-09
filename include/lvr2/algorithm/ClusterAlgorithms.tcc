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
 * ClusterAlgorithms.tcc
 *
 * @date 24.07.2017
 * @author Johan M. von Behren <johan@vonbehren.eu>
 */

#include <lvr2/algorithm/Planar.hpp>
#include <lvr2/algorithm/ContourAlgorithms.hpp>

namespace lvr2
{

template<typename BaseVecT>
void removeDanglingCluster(BaseMesh<BaseVecT>& mesh, size_t sizeThreshold)
{
    // Do cluster growing without a predicate, so cluster will consist of connected faces
    auto clusterSet = clusterGrowing(mesh, [](auto referenceFaceH, auto currentFaceH)
    {
        return true;
    });

    // Remove all faces in too small clusters
    for (auto clusterH: clusterSet)
    {
        auto cluster = clusterSet.getCluster(clusterH);
        if (cluster.handles.size() < sizeThreshold)
        {
            for (auto faceH: cluster.handles)
            {
                mesh.removeFace(faceH);
            }
        }
    }
}

template<typename BaseVecT>
vector<vector<VertexHandle>> findContours(
    BaseMesh<BaseVecT>& mesh,
    const ClusterBiMap<FaceHandle>& clusters,
    ClusterHandle clusterH
)
{
    auto cluster = clusters[clusterH];

    DenseVertexMap<bool> boundaryVertices(cluster.handles.size() * 3, false);
    vector<vector<VertexHandle>> allContours;
    // only used inside edge loop but initialized here to avoid heap allocations
    vector<VertexHandle> contour;

    for (auto faceH: cluster.handles)
    {
        for (auto edgeH: mesh.getEdgesOfFace(faceH))
        {
            auto faces = mesh.getFacesOfEdge(edgeH);
            auto otherFace = faces[0].unwrap();

            // switch face to get the other one of the edge if it exists
            if (faces[1] && otherFace == faceH) {
                otherFace = faces[1].unwrap();
            }

            // continue if other face is same cluster and is not an boundary edge
            if (faces[1] &&
                clusters.getClusterOf(otherFace) &&
                clusters.getClusterOf(otherFace).unwrap() == clusterH
                )
            {
                continue;
            }

            auto vertices = mesh.getVerticesOfEdge(edgeH);

            // edge already in another boundary of this cluster
            if (boundaryVertices[vertices[0]] || boundaryVertices[vertices[1]])
            {
                continue;
            }

            contour.clear();
            calcContourVertices(mesh, edgeH, contour, [clusters, clusterH](auto fH)
            {
                auto c = clusters.getClusterOf(fH);

                // return true if current face is in this cluster
                return c && c.unwrap() == clusterH;
            });

            allContours.push_back(contour);

            // mark all vertices we got back as visited
            for (auto vertexH: contour)
            {
                boundaryVertices[vertexH] = true;
            }
        }

    }

    return allContours;
}

} // namespace lvr2
