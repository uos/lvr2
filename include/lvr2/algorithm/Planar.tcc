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
#include <lvr2/util/VectorMap.hpp>
#include <lvr2/geometry/HalfEdgeMesh.hpp>
#include <lvr2/geometry/BoundingBox.hpp>
#include <lvr2/util/Debug.hpp>

using std::unordered_set;
using std::vector;
using std::max;
using std::log;

namespace lvr2
{

template<typename BaseVecT>
ClusterSet<FaceHandle> planarClusterGrowing(
    const BaseMesh<BaseVecT>& mesh,
    float minSinAngle
)
{
    ClusterSet<FaceHandle> clusters;
    FaceMap<bool> visited(mesh.numFaces(), false);

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
            auto referenceNormal = mesh.getFaceNormal(faceH);

            // Grow my cluster, groOW!
            while (!stack.empty())
            {
                auto currentFace = stack.back();
                stack.pop_back();

                // Check if the last faces from stack and starting face are in the "same" plane
                if (mesh.getFaceNormal(currentFace).dot(referenceNormal.asVector()) > minSinAngle)
                {
                    // The face is in the "same" plane as the starting face => add it to cluster
                    // and mark as visited.
                    clusters.addToCluster(cluster, currentFace);
                    visited[currentFace] = true;

                    // Find all unvisited neighbours of the current face and them to the stack
                    for (auto neighbour: mesh.getNeighboursOfFace(currentFace))
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
ClusterSet<FaceHandle> iterativePlanarClusterGrowing(
    BaseMesh<BaseVecT>& mesh,
    float minSinAngle,
    int numIterations,
    int minClusterSize
)
{
    ClusterSet<FaceHandle> clusters;

    // Iterate numIterations times
    for (int i = 0; i < numIterations; ++i)
    {
        // Generate clusters
        clusters = planarClusterGrowing(mesh, minSinAngle);

        // Calc regression planes
        auto planes = calcRegressionPlanes(mesh, clusters, minClusterSize);

        // Debug planes
        // std::stringstream ss;
        // ss << "debug" << i << ".ply";
        // debugPlanes(mesh, clusters, planes, ss.str(), 10000);
        // ss.str("");
        // ss << "debug-mesh" << i << ".ply";
        // writeDebugMesh(mesh, ss.str(), {0, 255, 0});

        // Drag vertices into planes
        dragToRegressionPlanes(mesh, clusters, planes);
    }

    return clusters;
}

template<typename BaseVecT>
ClusterMap<Plane<BaseVecT>> calcRegressionPlanes(
    const BaseMesh<BaseVecT>& mesh,
    const ClusterSet<FaceHandle>& clusters,
    int minClusterSize
)
{
    ClusterMap<Plane<BaseVecT>> planes;
    size_t defaultClusterThreshold = 10 * log(mesh.numFaces());
    size_t minClusterThresholdSize = max(static_cast<size_t>(minClusterSize), defaultClusterThreshold);

    // For all clusters in cluster set
    for (auto clusterH: clusters)
    {
        // Get current cluster
        auto cluster = clusters[clusterH];
        if (cluster.handles.size() > minClusterThresholdSize)
        {
            // Calc regression plane for current cluster and add to cluster map: cluster -> plane
            planes.insert(clusterH, calcRegressionPlane(mesh, cluster));
        }
    }

    return planes;
}

template<typename BaseVecT>
Plane<BaseVecT> calcRegressionPlane(
    const BaseMesh<BaseVecT>& mesh,
    const Cluster<FaceHandle>& cluster
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
        auto normal = mesh.getFaceNormal(faceH).asVector();

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
        for (auto vH: mesh.getVertexHandlesOfFace(faceH))
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
    const ClusterSet<FaceHandle>& clusters,
    const ClusterMap<Plane<BaseVecT>>& planes
)
{
    // For all clusters in cluster set
    for (auto clusterH: planes)
    {
        // Drag all vertices of current cluster into regression plane
        dragToRegressionPlane(mesh, clusters[clusterH], planes[clusterH]);
    }
}

template<typename BaseVecT>
void dragToRegressionPlane(
    BaseMesh<BaseVecT>& mesh,
    const Cluster<FaceHandle>& cluster,
    const Plane<BaseVecT>& plane
)
{
    for (auto faceH: cluster.handles)
    {
        for (auto vertexH: mesh.getVertexHandlesOfFace(faceH))
        {
            auto pos = mesh.getVertexPosition(vertexH);
            auto distance = plane.distance(pos);
            mesh.vertexPosition(vertexH) -= plane.normal.asVector() * distance;
        }
    }
}

template<typename BaseVecT>
void debugPlanes(
    const BaseMesh<BaseVecT>& mesh,
    const ClusterSet<FaceHandle>& clusters,
    const ClusterMap<Plane<BaseVecT>>& planes,
    string filename,
    size_t minClusterSize
)
{
    HalfEdgeMesh<BaseVecT> debugMesh;

    // For all clusters in cluster set
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

} // namespace lvr2
