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
 * ClusterAlgorithm.tcc
 *
 *  @date 20.07.2017
 *  @author Jan Philipp Vogtherr <jvogtherr@uni-osnabrueck.de>
 *  @author Kristin Schmidt <krschmidt@uni-osnabrueck.de>
 */

#include <lvr2/geometry/Cluster.hpp>
#include <lvr2/geometry/HalfEdgeMesh.hpp>

namespace lvr2
{


template<typename BaseVecT>
vector<Point<BaseVecT>> calculateAllContourVertices(
    ClusterHandle clusterH,
    HalfEdgeMesh<BaseVecT>& mesh,
    ClusterSet<FaceHandle>& clusterSet
)
{
    std::vector<VertexHandle> allContours;
    std::vector<Point<BaseVecT>> allContourVertices;

    auto cluster = clusterSet.getCluster(clusterH);

    // iterate all faces in cluster
    for (auto faceH : cluster.handles)
    {
        // find contours of each face
        std::vector<EdgeHandle> contours = mesh.getContourEdgesOfFaceDebug(
            faceH,
            [&](auto neighbourFaceH)
            {
                // pred must return true when faces are not in the same cluster
                return (clusterH != clusterSet.getClusterH(neighbourFaceH));
            }
        );

        // find all vertices from contour edges
        for (auto edgeH : contours)
        {
            for (auto vertexH : mesh.getVerticesOfEdge(edgeH))
            {
                allContourVertices.push_back(mesh.getVertexPosition(vertexH));
            }
        }
    }

    // remove duplicates from result
    std::sort(
        allContourVertices.begin(),
        allContourVertices.end(),
        [] ( const Point<BaseVecT>& lhs, const Point<BaseVecT>& rhs) {
            return lhs.x < rhs.x && lhs.y < rhs.y && lhs.z < rhs.z;
        }
    );
    auto comp =
        [] ( const Point<BaseVecT>& lhs, const Point<BaseVecT>& rhs) {
            return lhs == rhs;
        };
    allContourVertices.erase(
        std::unique(allContourVertices.begin(), allContourVertices.end(), comp),
        allContourVertices.end()
    );

    return allContourVertices;
}



template<typename BaseVecT>
vector<vector<VertexHandle>> calculateContour(
    ClusterHandle clusterH,
    HalfEdgeMesh<BaseVecT>& mesh,
    ClusterSet<FaceHandle>& clusterSet
)
{
    // nothing works

    vector<vector<VertexHandle>> result;

    auto cluster = clusterSet.getCluster(clusterH);

    size_t numFaces = cluster.handles.size();
    FaceMap<bool> visitedFaces(numFaces, false);
    EdgeMap<bool> visitedContourEdges(numFaces * 3, false); // TODO: use native hashmap


    // iterate all faces
    for (auto faceH : cluster.handles)
    {
        // mark current face as visited
        visitedFaces[faceH] = true;

        // find contours of face
        std::vector<EdgeHandle> contours = mesh.getContourEdgesOfFace(
            faceH,
            [&](auto neighbourFaceH)
            {
                // pred must return true when faces are not in the same cluster
                return (clusterH != clusterSet.getClusterH(neighbourFaceH));
            }
        );

        vector<VertexHandle> innerResult;

        // iterate all edges in contour of this face
        for (auto edgeH : contours)
        {
            auto currentEdgeH = edgeH;
            do
            {
                // if not visited yet
                if (!visitedContourEdges[currentEdgeH])
                {
                    // mark edge as visited
                    visitedContourEdges[currentEdgeH] = true;

                    auto edgeHVertices = mesh.getVerticesOfEdge(currentEdgeH);
                    // alle edges am vertex
                    auto targetEdges = mesh.getEdgesOfVertex(edgeHVertices[0]);

                    for (auto nextEdgeH : targetEdges)
                    {
                        // wenn nicht OG edge
                        if (mesh.getVerticesOfEdge(nextEdgeH)[0] != edgeHVertices[1])
                        {
                            auto faceOfEdgeHOptional = mesh.getFacesOfEdge(nextEdgeH)[0];
                            if (faceOfEdgeHOptional)
                            {
                                bool isContourFace = clusterH != clusterSet.getClusterH(faceOfEdgeHOptional.unwrap());
                                if (isContourFace)
                                {
                                    innerResult.push_back(edgeHVertices[0]);
                                    currentEdgeH = nextEdgeH;
                                }
                            }
                        }
                    }
                }
            } while(currentEdgeH != edgeH);
        }

        result.push_back(innerResult);
    }


    // unordered_set<VertexHandle> set;

    // std::vector<EdgeHandle> allContours;
    // auto cluster = clusterSet.getCluster(clusterH);

    // // iterate all faces in cluster
    // for (auto faceH : cluster.handles)
    // {
    //     // find contours of each face
    //     std::vector<EdgeHandle> contours = mesh.getContourEdgesOfFace(
    //         faceH,
    //         [&](auto neighbourFaceH)
    //         {
    //             // pred must return true when faces are not in the same cluster
    //             return (clusterH != clusterSet.getClusterH(neighbourFaceH));
    //         }
    //     );

    //     // find all vertices from contour edges
    //     for (auto edgeH : contours)
    //     {
    //         allContours.push_back(edgeH);
    //     }
    // }





    return result;

}


} // namespace lvr2
