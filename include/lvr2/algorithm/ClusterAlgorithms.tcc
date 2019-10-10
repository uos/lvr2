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
 * ClusterAlgorithms.tcc
 *
 * @date 24.07.2017
 * @author Johan M. von Behren <johan@vonbehren.eu>
 * @author Jan Philipp Vogtherr <jvogtherr@uni-osnabrueck.de>
 * @author Kristin Schmidt <krschmidt@uni-osnabrueck.de>
 * @author Rasmus Diederichsen <rdiederichse@uni-osnabrueck.de>
 */

#include "lvr2/algorithm/ContourAlgorithms.hpp"
#include "lvr2/geometry/BoundingBox.hpp"
#include "lvr2/geometry/HalfEdgeMesh.hpp"
#include "lvr2/util/Random.hpp"
#include "lvr2/geometry/Normal.hpp"
#include "lvr2/util/Debug.hpp"

#include "lvr2/io/Progress.hpp"
#include "lvr2/io/Timestamp.hpp"

#include <algorithm>
#include <complex>
#include <sstream>
#include <cmath>
#include <limits>
#include <unordered_set>

using std::unordered_set;
using std::max;
using std::log;

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
            if (faces[0] && faces[1])
            {
                auto otherFace = faces[0].unwrap();

                if (otherFace == faceH)
                {
                    otherFace = faces[1].unwrap();
                }

                // continue if other face is in same cluster
                if (clusters.getClusterOf(otherFace) &&
                    clusters.getClusterOf(otherFace).unwrap() == clusterH
                    )
                {
                    continue;
                }
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

template<typename BaseVecT>
vector<VertexHandle>
simplifyContour(const BaseMesh<BaseVecT>& mesh, const vector<VertexHandle>& contour, float threshold) {
    auto out = vector<VertexHandle>();

    // first point is always part of the simplified contour
    out.push_back(contour[0]);

    // current point
    auto p0 = mesh.getVertexPosition(contour[0]);
    // next point after first point
    auto p1 = mesh.getVertexPosition(contour[1]);

    // previous test point handle
    auto piH = contour[1];
    // current test point handle
    auto pjH = piH;

    for (size_t i = 2 ; i < contour.size(); ++i)
    {
        piH = pjH;
        pjH = contour[i];

        // current key vector / line to check against
        auto vec = p1 - p0;
        vec.normalize();

        // vector of next and after next point
        auto vec2 = mesh.getVertexPosition(pjH) - mesh.getVertexPosition(piH);
        vec2.normalize();

        if (vec.dot(vec2) >= threshold)
        {
            continue;
        }

        // found next point in line
        out.push_back(piH);

        // define new line to check against
        p0 = mesh.getVertexPosition(piH);
        p1 = mesh.getVertexPosition(pjH);
    }

    // also add last last point to simplified contour
    out.push_back(pjH);

    return out;
}


template<typename BaseVecT>
std::vector<VertexHandle> calculateClusterContourVertices(
    ClusterHandle clusterH,
    const BaseMesh<BaseVecT>& mesh,
    const ClusterBiMap<FaceHandle>& clusterBiMap
)
{
    std::unordered_set<VertexHandle> contourVertices;

    auto cluster = clusterBiMap.getCluster(clusterH);

    // iterate over all faces in cluster
    for (auto faceH : cluster.handles)
    {
        // get edges of each face
        const auto edgesOfFace = mesh.getEdgesOfFace(faceH);


        // check for each edge if it is a contour edge
        for (auto edgeH : edgesOfFace)
        {
            const auto faces = mesh.getFacesOfEdge(edgeH);
            int numFaces = 0;

            // count how many faces the edge has that are in the same cluster
            numFaces += faces[0] && clusterH == clusterBiMap.getClusterH(faces[0].unwrap()) ? 1 : 0;
            numFaces += faces[1] && clusterH == clusterBiMap.getClusterH(faces[1].unwrap()) ? 1 : 0;

            // if there is exactly one face, the edge is a contour edge
            if (numFaces == 1)
            {
                // add the vertices of contour edges to an unordered set, which
                // automatically doesn't add duplicates
                for (auto vertexH : mesh.getVerticesOfEdge(edgeH))
                {
                    contourVertices.insert(vertexH);
                }
            }
        }
    }

    return std::vector<VertexHandle>(contourVertices.begin(), contourVertices.end());
}

template<typename BaseVecT>
BoundingRectangle<typename BaseVecT::CoordType> calculateBoundingRectangle(
    const std::vector<VertexHandle>& contour,
    const BaseMesh<BaseVecT>& mesh,
    const Cluster<FaceHandle>& cluster,
    const FaceMap<Normal<typename BaseVecT::CoordType>>& normals,
    float texelSize,
    ClusterHandle clusterH
)
{

    // TODO error handling for texelSize = 0
    // TODO reasonable error handling necessary for empty contour vector
    if (contour.size() == 0)
    {
        cout << "Empty contour array." << endl;
    }
    int minArea = std::numeric_limits<int>::max();

    float bestMinA, bestMaxA, bestMinB, bestMaxB;
    BaseVecT bestBoundingAxisA, bestBoundingAxisB;

    // calculate regression plane for the cluster
    Plane<BaseVecT> regressionPlane = calcRegressionPlane(mesh, cluster, normals);

    // support vector for the plane
    BaseVecT supportVector = regressionPlane.project(mesh.getVertexPosition(contour[0]));

    // calculate two orthogonal vectors in the plane
    auto normal = regressionPlane.normal;
    auto pointInPlane = regressionPlane.project(mesh.getVertexPosition(contour[1]));
    auto boudningAxis1 = (pointInPlane - supportVector).cross(normal);
    boudningAxis1.normalize();

    BaseVecT boundingAxis2 = boudningAxis1.cross(normal);
    boundingAxis2.normalize();

    // const float pi = boost::math::constants::pi<float>(); // FIXME: doesnt seem to work with c++11
    const float pi = std::atan(1) * 4; // reasonable approximation for pi

    // resolution of iterative improvement steps for a fourth rotation
    const float delta = (pi / 2) / 90;

    for(float theta = 0; theta < M_PI / 2; theta += delta)
    {
        // rotate the bounding box
        boudningAxis1 = boudningAxis1 * cos(theta) + boundingAxis2 * sin(theta);
        boudningAxis1.normalize();
        boundingAxis2 = boudningAxis1.cross(normal);
        boundingAxis2.normalize();

        // FIXME
        // calculate hessian normal forms for both planes to which the distances will be calculated

        // assume each bounding box axis is the normal of a plane, then the dot
        // product with the support vector is the support vectors negative
        // distance to the plane given by the axis (n * sv = -p)
        // Note that in contrast to the usual plane equations, the plane
        // distance is missing the negative sign
        const float planeDist1 = boudningAxis1.dot(supportVector);
        const float planeDist2 = boundingAxis2.dot(supportVector);


        float minDistA = std::numeric_limits<float>::max();
        float maxDistA = std::numeric_limits<float>::lowest();
        float minDistB = std::numeric_limits<float>::max();
        float maxDistB = std::numeric_limits<float>::lowest();


        // calculate the bounding box

        for(const auto contourVertexH : contour)
        {
            // possible improvement: calculate the contourPoints only once before the for loop
            auto contourPoint = regressionPlane.project(mesh.getVertexPosition(contourVertexH));

            // use hessian plane form for distance calculation
            // note the negative sign of planeDist*, since the calculation above
            // actually computes the negative distance
            // calculate distance to plane1
            float distA = boudningAxis1.dot(contourPoint) - planeDist1;
            // calculate distance to plane2
            float distB = boundingAxis2.dot(contourPoint) - planeDist2;

            // memorize largest positive and negative distance to both planes
            if (distA > maxDistA)
            {
                maxDistA = distA;
            }
            if (distA < minDistA)
            {
                minDistA = distA;
            }
            if (distB > maxDistB)
            {
                maxDistB = distB;
            }
            if (distB < minDistB)
            {
                minDistB = distB;
            }
        }

        // calculate predicted number of texels for both dimesions
        int texelsX = std::ceil((maxDistA - minDistA) / texelSize);
        int texelsY = std::ceil((maxDistB - minDistB) / texelSize);

        // iterative improvement of the area
        if(texelsX * texelsY < minArea)
        {
            minArea           = texelsX * texelsY;
            bestMinA          = minDistA;
            bestMaxA          = maxDistA;
            bestMinB          = minDistB;
            bestMaxB          = maxDistB;
            bestBoundingAxisA = boudningAxis1;
            bestBoundingAxisB = boundingAxis2;
        }
    }

    return BoundingRectangle<typename BaseVecT::CoordType>(
        supportVector,
        bestBoundingAxisA,
        bestBoundingAxisB,
        normal,
        bestMinA,
        bestMaxA,
        bestMinB,
        bestMaxB
    );

}


template<typename BaseVecT, typename Pred>
ClusterBiMap<FaceHandle> clusterGrowing(const BaseMesh<BaseVecT>& mesh, Pred pred)
{
    ClusterBiMap<FaceHandle> clusters;
    DenseFaceMap<bool> visited(mesh.numFaces(), false);

    // This vector is only used later, but in order to avoid heap allocations
    // we will create this list here to retain the buffer.
    vector<FaceHandle> faceNeighbours;

    // Iterate over all faces
    for (auto faceH: mesh.faces())
    {
        // Check if face is in a cluster (i.e. we have not visited it)
        if (!visited[faceH])
        {
            // We found a face yet to be visited. Prepare things for growing.
            vector<FaceHandle> stack;
            stack.push_back(faceH);
            auto cluster = clusters.createCluster();

            // Grow my cluster, groOW!
            while (!stack.empty())
            {
                auto currentFace = stack.back();
                stack.pop_back();

                // Check if the last faces from stack and starting face match the criteria to join the same cluster
                if (!visited[currentFace] && pred(faceH, currentFace))
                {
                    // The face matched the criteria => add it to cluster and mark as visited.
                    clusters.addToCluster(cluster, currentFace);
                    visited[currentFace] = true;

                    // Find all unvisited neighbours of the current face and them to the stack
                    faceNeighbours.clear();
                    mesh.getNeighboursOfFace(currentFace, faceNeighbours);
                    for (auto neighbour: faceNeighbours)
                    {
                        if (!visited[neighbour])
                        {
                            stack.push_back(neighbour);
                        }
                    }
                }
            }
        }
    }

    return clusters;
}

template<typename BaseVecT>
ClusterBiMap<FaceHandle> planarClusterGrowing(
    const BaseMesh<BaseVecT>& mesh,
    const FaceMap<Normal<typename BaseVecT::CoordType>>& normals,
    float minSinAngle
)
{
    return clusterGrowing(mesh, [&](auto referenceFaceH, auto currentFaceH)
    {
        return normals[currentFaceH].dot(normals[referenceFaceH]) > minSinAngle;
    });
}

template<typename BaseVecT>
ClusterBiMap<FaceHandle> iterativePlanarClusterGrowing(
    BaseMesh<BaseVecT>& mesh,
    FaceMap<Normal<typename BaseVecT::CoordType>>& normals,
    float minSinAngle,
    int numIterations,
    int minClusterSize
)
{
    ClusterBiMap<FaceHandle> clusters;
    DenseClusterMap<Plane<BaseVecT>> planes;

    // Iterate numIterations times
    for (int i = 0; i < numIterations; ++i)
    {
        std::cout << timestamp << "Optimizing planes. Iterations "
                  << i << " / " << numIterations << std::endl;
        // Generate clusters
        clusters = planarClusterGrowing(mesh, normals, minSinAngle);

        // Calc regression planes
        planes = calcRegressionPlanes(mesh, clusters, normals, minClusterSize);

        // Drag vertices into planes
        dragToRegressionPlanes(mesh, clusters, planes, normals);
    }

    optimizePlaneIntersections(mesh, clusters, planes);

    return clusters;
}


template<typename BaseVecT>
ClusterBiMap<FaceHandle> iterativePlanarClusterGrowingRANSAC(
    BaseMesh<BaseVecT>& mesh,
    FaceMap<Normal<typename BaseVecT::CoordType>>& normals,
    float minSinAngle,
    int numIterations,
    int minClusterSize,
    int ransacIterations,
    int ransacSamples
)
{
    ClusterBiMap<FaceHandle> clusters;
    DenseClusterMap<Plane<BaseVecT>> planes;

    // Iterate numIterations times
    for (int i = 0; i < numIterations; ++i)
    {
        std::cout << timestamp << "Optimizing planes. Iterations "
                  << i << " / " << numIterations << std::endl;
        // Generate clusters
        clusters = planarClusterGrowing(mesh, normals, minSinAngle);

        // Calc regression planes
        planes = calcRegressionPlanesRANSAC(mesh,
                    clusters,
                    normals,
                    minClusterSize,
                    ransacIterations,
                    ransacSamples);

        // Drag vertices into planes
        dragToRegressionPlanes(mesh, clusters, planes, normals);
    }

    optimizePlaneIntersections(mesh, clusters, planes);

    return clusters;
}

template<typename BaseVecT>
DenseClusterMap<Plane<BaseVecT>> calcRegressionPlanes(
    const BaseMesh<BaseVecT>& mesh,
    const ClusterBiMap<FaceHandle>& clusters,
    const FaceMap<Normal<typename BaseVecT::CoordType>>& normals,
    int minClusterSize
)
{
    DenseClusterMap<Plane<BaseVecT>> planes;
    size_t defaultClusterThreshold = 10 * log(mesh.numFaces());
    size_t minClusterThresholdSize = max(static_cast<size_t>(minClusterSize), defaultClusterThreshold);

    // For all clusters in cluster map
    for (auto clusterH: clusters)
    {
        // Get current cluster
        auto cluster = clusters[clusterH];
        if (cluster.handles.size() > minClusterThresholdSize)
        {
            // Calc regression plane for current cluster and add to cluster map: cluster -> plane
            planes.insert(clusterH, calcRegressionPlanePCA(mesh, cluster, normals));
        }
    }

    return planes;
}

template<typename BaseVecT>
DenseClusterMap<Plane<BaseVecT>> calcRegressionPlanesRANSAC(
    const BaseMesh<BaseVecT>& mesh,
    const ClusterBiMap<FaceHandle>& clusters,
    const FaceMap<Normal<typename BaseVecT::CoordType>>& normals,
    int minClusterSize,
    int iterations,
    int samples
)
{
    DenseClusterMap<Plane<BaseVecT>> planes;
    size_t defaultClusterThreshold = 10 * log(mesh.numFaces());
    size_t minClusterThresholdSize = max(static_cast<size_t>(minClusterSize), defaultClusterThreshold);

    // For all clusters in cluster map
    for (auto clusterH: clusters)
    {
        // Get current cluster
        auto cluster = clusters[clusterH];
        if (cluster.handles.size() > minClusterThresholdSize)
        {
            // Calc regression plane for current cluster and add to cluster map: cluster -> plane
            planes.insert(clusterH, calcRegressionPlaneRANSAC(mesh, cluster, normals, iterations, samples));
        }
    }

    return planes;
}

template<typename BaseVecT>
Plane<BaseVecT> calcRegressionPlane(
    const BaseMesh<BaseVecT>& mesh,
    const Cluster<FaceHandle>& cluster,
    const FaceMap<Normal<typename BaseVecT::CoordType>>& normals
)
{
    // Calc normal of plane
    size_t countFirst = 0;
    size_t countSecond = 0;

    BaseVecT vectorFirst;
    BaseVecT vectorSecond;

    // Average Normals for both normal directions
    for (auto faceH: cluster.handles)
    {
        auto normal = normals[faceH];

        // If first iteration
        if (countFirst == 0)
        {
            vectorFirst = normal;
            ++countFirst;
        }
        else
        {
            // Split normals by orientation
            auto scalar = normal.dot(vectorFirst);
            if (scalar > 0)
            {
                vectorFirst += normal;
                ++countFirst;
            }
            else
            {
                vectorSecond += normal;
                ++countSecond;
            }
        }
    }

    // Flip normals to same direction
    BaseVecT vector;
    if (countFirst > countSecond)
    {
        vector = vectorFirst;
        vector -= vectorSecond;
    }
    else
    {
        vector = vectorSecond;
        vector -= vectorFirst;
    }

    Normal<typename BaseVecT::CoordType> normal(vector);
    Plane<BaseVecT> plane;
    plane.normal = normal;
    plane.pos = mesh.getVertexPositionsOfFace(cluster.handles[0])[0];

    // Calc average distance from plane to all points
    float planeDistance = 0;
    unordered_set<VertexHandle> vertices;

    // Iterate over all faces in cluster to get all vertices in plane
    for (auto faceH: cluster.handles)
    {
        // Iterate over all vertices of current face
        for (auto vH: mesh.getVerticesOfFace(faceH))
        {
            // If current vertex is not visited, add distance
            if (!vertices.count(vH))
            {
                vertices.insert(vH);
                planeDistance += plane.distance(mesh.getVertexPosition(vH));
            }
        }
    }
    float avgDistance = planeDistance / vertices.size();

    // Move pos of plane to best fit
    plane.pos += (plane.normal * avgDistance);
    return plane;
}


template<typename BaseVecT>
Plane<BaseVecT> calcRegressionPlaneRANSAC(
    const BaseMesh<BaseVecT>& mesh,
    const Cluster<FaceHandle>& cluster,
    const FaceMap<Normal<typename BaseVecT::CoordType>>& normals,
    const int num_iterations,
    const int num_samples
)
{
    float error_limit = 0.01; // dynamically voxelsize / 100
    Plane<BaseVecT> best_plane;
    int best_inlier = 0;
    

    // RANSAC:
    // - select vertex + normal randomly -> plane
    // - count number of vertices within a predefined error range

    // 1) collect all vertices used for error computation
    //   + determine average edge length for automatic error thresh
    float avg_dist = 0.0;
    int num_edges = 0;
    unordered_set<VertexHandle> vertices;
    for (auto faceH: cluster.handles)
    {
        // Iterate over all vertices of current face
        boost::optional<VertexHandle> vHlast;
        for (auto vH: mesh.getVerticesOfFace(faceH))
        {
            // If current vertex is not visited, add distance
            if (!vertices.count(vH))
            {
                vertices.insert(vH);
            }

            if(vHlast)
            {
                avg_dist += mesh.getVertexPosition(*vHlast).distance(mesh.getVertexPosition(vH));
                num_edges++;
            }

            vHlast = vH;
        }
    }
    avg_dist /= static_cast<float>(num_edges);

    error_limit *= avg_dist;
    
    const size_t num_cluster_vertices = vertices.size();
    const size_t num_cluster_faces = cluster.size();

    for(int i=0; i<num_iterations; i++)
    {
        Plane<BaseVecT> plane;
        plane.pos.x = 0.0;
        plane.pos.y = 0.0;
        plane.pos.z = 0.0;
        plane.normal.x = 0.0;
        plane.normal.y = 0.0;
        plane.normal.z = 0.0;
        
        // build avg plane of RANSAC samples
        for(int j=0; j<num_samples; j++)
        {
            const FaceHandle& faceHandle = cluster.handles[rand() % num_cluster_faces];
            plane.pos += mesh.getVertexPositionsOfFace(faceHandle)[rand() % 3];
            plane.normal += normals[faceHandle];
        }

        plane.pos /= static_cast<float>(num_samples);
        plane.normal.normalize();
        

        // calulate inlier
        int inlier = 0;
        for(auto vertexH : vertices)
        {
            const float current_dist = plane.distance(mesh.getVertexPosition(vertexH));
            if(fabs(current_dist) < error_limit)
            {
                inlier++;
            }
        }

        if(inlier > best_inlier)
        {
            best_inlier = inlier;
            best_plane = plane;
        }
    }

    return best_plane;
}


template<typename BaseVecT>
Plane<BaseVecT> calcRegressionPlanePCA(
    const BaseMesh<BaseVecT>& mesh,
    const Cluster<FaceHandle>& cluster,
    const FaceMap<Normal<typename BaseVecT::CoordType>>& normals,
    const int num_iterations,
    const int num_samples
)
{
    Plane<BaseVecT> plane;

    Eigen::Vector3d center(0,0,0);

    unordered_set<VertexHandle> vertices;
    for (auto faceH: cluster.handles)
    {
        // Iterate over all vertices of current face
        for (auto vH: mesh.getVerticesOfFace(faceH))
        {
            // If current vertex is not visited, add distance
            if (!vertices.count(vH))
            {
                vertices.insert(vH);
                const BaseVecT& pos = mesh.getVertexPosition(vH);
                center(0) += static_cast<double>(pos.x);
                center(1) += static_cast<double>(pos.y);
                center(2) += static_cast<double>(pos.z);
            }
        }
    }

    center /= static_cast<double>(vertices.size());

    Eigen::Matrix3Xd data(3, vertices.size());
    int current_vertex = 0;

    for(auto vH : vertices)
    {
        const BaseVecT& pos = mesh.getVertexPosition(vH);
        data.coeffRef(0, current_vertex) = static_cast<double>(pos.x);
        data.coeffRef(1, current_vertex) = static_cast<double>(pos.y);
        data.coeffRef(2, current_vertex) = static_cast<double>(pos.z);

        current_vertex++;
    }
    
    const Eigen::Matrix3Xd centered = data.array().colwise() - center.array();
    const Eigen::MatrixXd cov = (centered * centered.transpose()) / centered.cols();
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(cov);
    Eigen::MatrixXd evecs = eig.eigenvectors();
    const Eigen::Vector3d x = evecs.col(1);
    const Eigen::Vector3d y = evecs.col(2);
    Eigen::Vector3d n = x.cross(y).normalized();

    plane.pos.x = center(0);
    plane.pos.y = center(1);
    plane.pos.z = center(2);
    plane.normal.x = n(0);
    plane.normal.y = n(1);
    plane.normal.z = n(2);

    return plane;
}

template<typename BaseVecT>
void dragToRegressionPlanes(
    BaseMesh<BaseVecT>& mesh,
    const ClusterBiMap<FaceHandle>& clusters,
    const ClusterMap<Plane<BaseVecT>>& planes,
    FaceMap<Normal<typename BaseVecT::CoordType>>& normals
)
{
    // For all clusters in cluster map
    for (auto clusterH: planes)
    {
        // Drag all vertices of current cluster into regression plane
        dragToRegressionPlane(mesh, clusters[clusterH], planes[clusterH], normals);
    }
}

template<typename BaseVecT>
void dragToRegressionPlane(
    BaseMesh<BaseVecT>& mesh,
    const Cluster<FaceHandle>& cluster,
    const Plane<BaseVecT>& plane,
    FaceMap<Normal<typename BaseVecT::CoordType>>& normals
)
{
    for (auto faceH: cluster.handles)
    {
        for (auto vertexH: mesh.getVerticesOfFace(faceH))
        {
            auto pos = mesh.getVertexPosition(vertexH);
            auto distance = plane.distance(pos);
            mesh.getVertexPosition(vertexH) -= plane.normal * distance;
        }
        normals[faceH] = plane.normal;
    }
}

template<typename BaseVecT>
void optimizePlaneIntersections(
    BaseMesh<BaseVecT>& mesh,
    const ClusterBiMap<FaceHandle>& clusters,
    const ClusterMap<Plane<BaseVecT>>& planes
)
{
    // Status message for mesh generation
    string comment = timestamp.getElapsedTime() + "Optimizing plane intersections ";
    ProgressBar progress(planes.numValues(), comment);

    // iterate over all planes
    for (auto it = planes.begin(); it != planes.end(); ++it)
    {
        auto clusterH = *it;

        // only iterate over distinct pairs of planes, e.g. the following planes of the current one
        auto itInner = it;
        ++itInner;
        for (; itInner != planes.end(); ++itInner)
        {
            auto clusterInnerH = *itInner;

            auto& plane1 = planes[clusterH];
            auto& plane2 = planes[clusterInnerH];

            // do not improve almost parallel cluster
            float normalDot = plane1.normal.dot(plane2.normal);
            if (fabs(normalDot) < 0.9)
            {
                auto intersection = plane1.intersect(plane2);

                dragOntoIntersection(mesh, clusters, clusterH, clusterInnerH, intersection);
                dragOntoIntersection(mesh, clusters, clusterInnerH, clusterH, intersection);
            }
        }

        ++progress;
    }

    if(!timestamp.isQuiet())
        cout << endl;
}

template<typename BaseVecT>
void dragOntoIntersection(
    BaseMesh<BaseVecT>& mesh,
    const ClusterBiMap<FaceHandle>& clusters,
    const ClusterHandle& clusterH,
    const ClusterHandle& neighbourClusterH,
    const Line<BaseVecT>& intersection
)
{
    for (auto faceH: clusters[clusterH].handles)
    {
        for (auto edgeH: mesh.getEdgesOfFace(faceH))
        {
            auto facesOfEdge = mesh.getFacesOfEdge(edgeH);

            // check if edge is real neighbour of second plane
            if ((facesOfEdge[0] && clusters.getClusterOf(facesOfEdge[0].unwrap()) == neighbourClusterH)
                || (facesOfEdge[1] && clusters.getClusterOf(facesOfEdge[1].unwrap()) == neighbourClusterH))
            {
                auto vertices = mesh.getVerticesOfEdge(edgeH);

                auto& v1 = mesh.getVertexPosition(vertices[0]);
                auto& v2 = mesh.getVertexPosition(vertices[1]);

                // project both vertices of the edge into the intersection
                v1 = intersection.project(v1);
                v2 = intersection.project(v2);
            }
        }

    }
}

template<typename BaseVecT>
void debugPlanes(
    const BaseMesh<BaseVecT>& mesh,
    const ClusterBiMap<FaceHandle>& clusters,
    const ClusterMap<Plane<BaseVecT>>& planes,
    string filename,
    size_t minClusterSize
)
{
    HalfEdgeMesh<BaseVecT> debugMesh;

    // For all clusters in cluster map
    for (auto clusterH: planes)
    {
        auto cluster = clusters[clusterH];
        auto plane = planes[clusterH];

        if (cluster.handles.size() < minClusterSize)
        {
            continue;
        }

        // Get bounding box for cluster
        BoundingBox<BaseVecT> bBox;
        for (auto faceH: cluster.handles)
        {
            auto vertices = mesh.getVertexPositionsOfFace(faceH);
            for (auto vertex: vertices)
            {
                bBox.expand(vertex);
            }
        }

        // Get intersection with bounding box and plane
        auto centroid = plane.project(bBox.getCentroid());
        auto point = plane.project(bBox.getMin());
        if (centroid == point)
        {
            point = plane.project(bBox.getMax());
        }
        auto tangent1 = Normal<typename BaseVecT::CoordType>(point - centroid);
        auto tangent2 = Normal<typename BaseVecT::CoordType>(plane.normal.cross(tangent1));

        auto v1 = plane.pos + tangent1 * bBox.getLongestSide();
        auto v2 = plane.pos - tangent1 * bBox.getLongestSide();
        auto v3 = plane.pos + tangent2 * bBox.getLongestSide();
        auto v4 = plane.pos - tangent2 * bBox.getLongestSide();

        // Add intersection plane to mesh
        auto vH1 = debugMesh.addVertex(v1);
        auto vH2 = debugMesh.addVertex(v2);
        auto vH3 = debugMesh.addVertex(v3);
        auto vH4 = debugMesh.addVertex(v4);

        debugMesh.addFace(vH1, vH3, vH2);
        debugMesh.addFace(vH1, vH2, vH4);
    }

    // Save debug mesh
    writeDebugMesh(debugMesh, filename);
}

template<typename BaseVecT>
void deleteSmallPlanarCluster(
    BaseMesh<BaseVecT>& mesh,
    ClusterBiMap<FaceHandle>& clusters,
    size_t smallClusterThreshold
)
{
    for (auto cH: clusters)
    {
        auto cluster = clusters.getCluster(cH);
        if (cluster.size() < smallClusterThreshold)
        {
            for (auto fH: cluster)
            {
                mesh.removeFace(fH);
            }
            clusters.removeCluster(cH);
        }
    }
}

} // namespace lvr2
