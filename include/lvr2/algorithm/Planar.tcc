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
 * Planar.tcc
 *
 *  @date 14.07.2017
 *  @author Johan M. von Behren <johan@vonbehren.eu>
 */

#include <limits>
#include <algorithm>
#include <complex>
#include <vector>
#include <sstream>
#include <unordered_set>

#include <lvr2/util/Random.hpp>
#include <lvr2/geometry/Normal.hpp>
#include <lvr2/geometry/HalfEdgeMesh.hpp>
#include <lvr2/geometry/BoundingBox.hpp>
#include <lvr2/util/Debug.hpp>

#include <lvr/io/Progress.hpp>

using std::unordered_set;
using std::vector;
using std::max;
using std::log;

namespace lvr2
{

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
    const FaceMap<Normal<BaseVecT>>& normals,
    float minSinAngle
)
{
    return clusterGrowing(mesh, [&](auto referenceFaceH, auto currentFaceH)
    {
        return normals[currentFaceH].dot(normals[referenceFaceH].asVector()) > minSinAngle;
    });
}

template<typename BaseVecT>
ClusterBiMap<FaceHandle> iterativePlanarClusterGrowing(
    BaseMesh<BaseVecT>& mesh,
    FaceMap<Normal<BaseVecT>>& normals,
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
        // Generate clusters
        clusters = planarClusterGrowing(mesh, normals, minSinAngle);

        // Calc regression planes
        planes = calcRegressionPlanes(mesh, clusters, normals, minClusterSize);

        // Debug planes
        // std::stringstream ss;
        // ss << "debug" << i << ".ply";
        // debugPlanes(mesh, clusters, planes, ss.str(), 10000);
        // ss.str("");
        // ss << "debug-mesh" << i << ".ply";
        // writeDebugMesh(mesh, ss.str(), {0, 255, 0});

        // Drag vertices into planes
        dragToRegressionPlanes(mesh, clusters, planes, normals);
    }

    optimizePlaneIntersections(mesh, clusters, planes, normals);

    return clusters;
}

template<typename BaseVecT>
DenseClusterMap<Plane<BaseVecT>> calcRegressionPlanes(
    const BaseMesh<BaseVecT>& mesh,
    const ClusterBiMap<FaceHandle>& clusters,
    const FaceMap<Normal<BaseVecT>>& normals,
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
            planes.insert(clusterH, calcRegressionPlane(mesh, cluster, normals));
        }
    }

    return planes;
}

template<typename BaseVecT>
Plane<BaseVecT> calcRegressionPlane(
    const BaseMesh<BaseVecT>& mesh,
    const Cluster<FaceHandle>& cluster,
    const FaceMap<Normal<BaseVecT>>& normals
)
{
    // Calc normal of plane
    size_t countFirst = 0;
    size_t countSecond = 0;

    Vector<BaseVecT> vectorFirst;
    Vector<BaseVecT> vectorSecond;

    // Average Normals for both normal directions
    for (auto faceH: cluster.handles)
    {
        auto normal = normals[faceH].asVector();

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
    Vector<BaseVecT> vector;
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

    Normal<BaseVecT> normal(vector);
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
    plane.pos += (plane.normal.asVector() * avgDistance);
    return plane;
}

template<typename BaseVecT>
void dragToRegressionPlanes(
    BaseMesh<BaseVecT>& mesh,
    const ClusterBiMap<FaceHandle>& clusters,
    const ClusterMap<Plane<BaseVecT>>& planes,
    FaceMap<Normal<BaseVecT>>& normals
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
    FaceMap<Normal<BaseVecT>>& normals
)
{
    for (auto faceH: cluster.handles)
    {
        for (auto vertexH: mesh.getVerticesOfFace(faceH))
        {
            auto pos = mesh.getVertexPosition(vertexH);
            auto distance = plane.distance(pos);
            mesh.getVertexPosition(vertexH) -= plane.normal.asVector() * distance;
        }
        normals[faceH] = plane.normal;
    }
}

template<typename BaseVecT>
void optimizePlaneIntersections(
    BaseMesh<BaseVecT> &mesh,
    const ClusterBiMap<FaceHandle> &clusters,
    const ClusterMap<Plane<BaseVecT>> &planes,
    FaceMap<Normal<BaseVecT>> &normals
)
{
    // Status message for mesh generation
    string comment = lvr::timestamp.getElapsedTime() + "Optimizing plane intersections ";
    lvr::ProgressBar progress(planes.numValues(), comment);

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

            auto &plane1 = planes[clusterH];
            auto &plane2 = planes[clusterInnerH];

            // do not improve almost parallel cluster
            float normalDot = plane1.normal.dot(plane2.normal.asVector());
            if (fabs(normalDot) < 0.9)
            {
                float d1 = plane1.normal.dot(plane1.pos);
                float d2 = plane2.normal.dot(plane2.pos);

                // TODO refactor intersection line into separate class
                // TODO move intersection calculation into plane method
                // direction of the intersection line
                auto direction = plane1.normal.cross(plane2.normal.asVector());
                // calculate the position of the intersection line
                auto p = (plane2.normal.asVector() * d1 - plane1.normal.asVector() * d2).cross(direction) * (1 / (direction.dot(direction)));

                dragOntoIntersection(mesh, clusters, clusterH, clusterInnerH, direction, p);
                dragOntoIntersection(mesh, clusters, clusterInnerH, clusterH, direction, p);
            }
        }

        ++progress;
    }

    if(!lvr::timestamp.isQuiet())
        cout << endl;
}

template<typename BaseVecT>
void dragOntoIntersection(
    BaseMesh<BaseVecT>& mesh,
    const ClusterBiMap<FaceHandle>& clusters,
    const ClusterHandle& clusterH,
    const ClusterHandle& neighbourClusterH,
    const Vector<BaseVecT>& direction,
    const Vector<BaseVecT>& p
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
                v1 = p + direction * ((v1 - p).dot(direction) / direction.length2());
                v2 = p + direction * ((v2 - p).dot(direction) / direction.length2());
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
        auto tangent1 = Normal<BaseVecT>(point - centroid);
        auto tangent2 = Normal<BaseVecT>(plane.normal.asVector().cross(tangent1.asVector()));

        auto v1 = plane.pos + tangent1.asVector() * bBox.getLongestSide();
        auto v2 = plane.pos - tangent1.asVector() * bBox.getLongestSide();
        auto v3 = plane.pos + tangent2.asVector() * bBox.getLongestSide();
        auto v4 = plane.pos - tangent2.asVector() * bBox.getLongestSide();

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
