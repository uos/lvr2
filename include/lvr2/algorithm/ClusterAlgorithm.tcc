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

#include <boost/math/constants/constants.hpp>
#include <cmath>
#include <limits>

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

template<typename BaseVecT>
BoundingRectangle<BaseVecT> calculateBoundingBox(
    const std::vector<Point<BaseVecT>> contour,
    const BaseMesh<BaseVecT>& mesh,
    const Cluster<FaceHandle>& cluster,
    const FaceMap<Normal<BaseVecT>>& normals,
    float texelSize
)
{
    // TODO reasonable error handling necessary for empty contour vector
    if (contour.size() == 0)
    {
        cout << "Empty contour array." << endl;
    }
    int minArea = std::numeric_limits<int>::max();

    float bestMinA, bestMaxA, bestMinB, bestMaxB;
    Vector<BaseVecT> bestVec1, bestVec2;

    // calculate regression plane for the cluster
    Plane<BaseVecT> regressionPlane = calcRegressionPlane(mesh, cluster, normals);

    // support vector for the plane
    Vector<BaseVecT> supportVector = regressionPlane.pos.asVector();

    // calculate two orthogonal vectors in the plane
    auto normal = regressionPlane.normal;
    auto vec1 = normal.cross(Vector<BaseVecT>(-normal.getY(), normal.getX(), 0) + normal.asVector());
    Vector<BaseVecT> vec2 = normal.cross(vec1);

    const float pi = boost::math::constants::pi<float>();

    // resolution of iterative improvement steps for a fourth rotation
    float delta = (pi / 2) / 90;

    for(float theta = 0; theta < M_PI / 2; theta += delta)
    {
        // rotate the bounding box
        vec1 = vec1 * cos(theta) + vec2 * sin(theta);
        vec2 = vec1.cross(normal.asVector());

        // calculate hessian normal forms for both planes to which the distances will be calculated
        Normal<BaseVecT> planeNormal1 = (supportVector.dot(vec1) >= 0)? vec1.normalized() : -vec1.normalized();
        float planeDist1 = planeNormal1.dot(supportVector);
        Normal<BaseVecT> planeNormal2 = (supportVector.dot(vec2) >= 0)? vec2.normalized() : -vec2.normalized();
        float planeDist2 = planeNormal2.dot(supportVector);

        float minA = std::numeric_limits<float>::max();
        float maxA = std::numeric_limits<float>::lowest();
        float minB = std::numeric_limits<float>::max();
        float maxB = std::numeric_limits<float>::lowest();


        // calculate the bounding box

        for(auto contourPoint: contour)
        {
            // calculate distance to plane1
            float dist1 = planeNormal1.dot(contourPoint) - planeDist1;
            // calculate distance to plane2
            float dist2 = planeNormal2.dot(contourPoint) - planeDist2;

            float a = dist1;
            float b = dist2;

            // memorize largest positive and negative distance to both planes
            if (a > maxA)
            {
                maxA = a;
            }
            if (a < minA)
            {
                minA = a;
            }
            if (b > maxB)
            {
                maxB = b;
            }
            if (b < minB)
            {
                minB = b;
            }
        }

        // calculate predicted number of texels for both dimesions
        int texelsX = std::ceil((maxA - minA) / texelSize);
        int texelsY = std::ceil((maxB - minB) / texelSize);

        //iterative improvement of the area
        if(texelsX * texelsY < minArea)
        {
            minArea = texelsX * texelsY;
            bestMinA = minA;
            bestMaxA = maxA;
            bestMinB = minB;
            bestMaxB = maxB;
            bestVec1 = vec1;
            bestVec2 = vec2;
        }
    }

    return BoundingRectangle<BaseVecT>(supportVector, vec1, vec2, normal, bestMinA, bestMaxA, bestMinB, bestMaxB);

}


} // namespace lvr2
