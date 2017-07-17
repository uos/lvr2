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

#include <limits>
#include <algorithm>
#include <complex>
#include <vector>

#include <lvr2/util/Random.hpp>
#include <lvr2/geometry/Normal.hpp>
#include <lvr2/util/VectorMap.hpp>

using std::vector;
using std::max;
using std::log;

namespace lvr2
{

template<typename BaseVecT>
ClusterSet<FaceHandle> planarClusterGrowing(const BaseMesh<BaseVecT>& mesh, float minSinAngle)
{
    ClusterSet<FaceHandle> clusterSet;
    FaceMap<bool> visited(mesh.numFaces(), false);

    // Iterate over all faces
    for (auto faceH: mesh.faces())
    {
        // Check if face is in a cluster (e.g. we have not visited it)
        if (!visited[faceH])
        {
            // We found a face yet to be visited. Prepare things for growing.
            vector<FaceHandle> stack;
            stack.push_back(faceH);
            auto cluster = clusterSet.createCluster();
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
                    clusterSet.addToCluster(cluster, currentFace);
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

    return clusterSet;
}

template<typename BaseVecT>
ClusterSet<FaceHandle>
iterativePlanarClusterGrowing(
    BaseMesh<BaseVecT>& mesh,
    float minSinAngle,
    int numIterations,
    int minRegionSize
)
{
    ClusterSet<FaceHandle> clusterSet;

    // Iterate numIterations times
    for (int i = 0; i < numIterations; ++i)
    {
        // Generate clusters
        clusterSet = planarClusterGrowing(mesh, minSinAngle);

        // Calc regression planes
        auto planes = calcRegressionPlanes(mesh, clusterSet, minRegionSize);

        // Drag vertices into planes
        dragToRegressionPlanes(mesh, clusterSet, planes);
    }

    return clusterSet;
}

template<typename BaseVecT>
ClusterMap<Plane<BaseVecT>>
    calcRegressionPlanes(
        const BaseMesh<BaseVecT>& mesh,
        const ClusterSet<FaceHandle>& clusterSet,
        int minRegionSize
    )
{
    ClusterMap<Plane<BaseVecT>> clusterMap;
    size_t defaultRegionThreshold = 10 * log(mesh.numFaces());
    size_t minRegionThresholdSize = max(static_cast<size_t>(minRegionSize), defaultRegionThreshold);

    // For all clusters in cluster set
    for (auto clusterH: clusterSet)
    {
        if (clusterSet[clusterH].handles.size() > minRegionThresholdSize)
        {
            // Calc regression plane for current cluster and add to cluster map: cluster -> plane
            clusterMap.insert(clusterH, calcRegressionPlane(mesh, clusterSet[clusterH]));
        }
    }

    return clusterMap;
}

template<typename BaseVecT>
Plane<BaseVecT> calcRegressionPlane(const BaseMesh<BaseVecT>& mesh, const Cluster<FaceHandle>& cluster)
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
    size_t countVertices = 0;
    for (auto faceH: cluster.handles)
    {
        // TODO: fix double point usage
        auto points = mesh.getVertexPositionsOfFace(faceH);
        for (auto point: points)
        {
            planeDistance += plane.distance(point);
            ++countVertices;
        }
    }
    float avgDistance = planeDistance / countVertices;

    // Move pos of plane to best fit
    plane.pos += (plane.normal.asVector() * avgDistance);
    return plane;
}

template<typename BaseVecT>
void dragToRegressionPlanes(
    BaseMesh<BaseVecT>& mesh,
    const ClusterSet<FaceHandle>& clusterSet,
    const ClusterMap<Plane<BaseVecT>>& clusterMap
)
{
    // For all clusters in cluster set
    for (auto clusterH: clusterMap)
    {
        // Drag all vertices of current cluster into regression plane
        dragToRegressionPlane(mesh, clusterSet[clusterH], clusterMap[clusterH]);
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

} // namespace lvr2
