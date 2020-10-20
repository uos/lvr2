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
 * GeometryAlgorithms.tcc
 */

#include <algorithm>
#include <limits>
#include <queue>
#include <set>

#include "lvr2/attrmaps/AttrMaps.hpp"
#include "lvr2/io/Progress.hpp"

namespace lvr2
{

template <typename BaseVecT>
void calcVertexLocalNeighborhood(
    const BaseMesh<BaseVecT> &mesh,
    VertexHandle vH,
    double radius,
    vector<VertexHandle> &neighborsOut)
{
    visitLocalNeighborhoodOfVertex(mesh, vH, radius, [&](auto newVH) {
        neighborsOut.push_back(newVH);
    });
}

template <typename BaseVecT, typename VisitorF>
void visitLocalVertexNeighborhood(
    const BaseMesh<BaseVecT> &mesh,
    std::set<VertexHandle> &invalid,
    VertexHandle vH,
    double radius,
    VisitorF visitor)
{
    // Prepare values for the radius test
    auto vPos = mesh.getVertexPosition(vH);
    const double radiusSquared = radius * radius;

    // Store the vertices we want to expand. We reserve memory for 8 vertices
    // already, since it's rather likely to have at least that many vertices
    // in the stack. In the beginning, the stack only contains the original
    // vertex we were given.
    vector<VertexHandle> stack;
    stack.reserve(8);
    stack.push_back(vH);

    // In this map we store whether or not we have already visited a vertex,
    // where visiting means: calling the visitor with it and pushing it on
    // the stack of vertices we still need to expand.
    //
    // It would be more appropriate to use a set instead of a map, but there
    // are no attribute-sets...
    SparseVertexMap<bool> visited(8, false);
    visited.insert(vH, true);

    // This vector is later used to store the neighbors of a vertex. It's
    // created here to reduce the amount of heap allocations.
    vector<VertexHandle> directNeighbors;

    // As long as there are vertices we want to expand...
    while (!stack.empty())
    {
        // Get the next vertex and remove it from the stack.
        auto curVH = stack.back();
        stack.pop_back();

        // Expand current vertex: add visit its direct neighbors.
        directNeighbors.clear();
        try
        {
            mesh.getNeighboursOfVertex(curVH, directNeighbors);
        }
        catch (lvr2::PanicException exception)
        {
            invalid.insert(curVH);
        }
        for (auto newVH : directNeighbors)
        {
            // If this vertex is within the radius of the original vertex, we
            // want to visit it later, thus pushing it onto the stack. But we
            // only do that if we haven't visited the vertex before. We use
            // `containsKey` here because we only insert `true` anyway.
            auto distSquared = mesh.getVertexPosition(newVH).squaredDistanceFrom(vPos);
            if (!visited.containsKey(newVH) && distSquared < radiusSquared)
            {
                visitor(newVH);
                stack.push_back(newVH);
                visited.insert(newVH, true);
            }
        }
    }
}

template <typename BaseVecT>
DenseVertexMap<float> calcVertexHeightDifferences(const BaseMesh<BaseVecT> &mesh, double radius)
{
    // We create a map to store a height-diff for each vertex. We preallocate
    // memory for all vertices. This is not only an optimization, but more
    // importantly to avoid multithreading crashes. The parallelized loop
    // further down inserts into this data structure; if there is no free
    // memory left, it will reallocate which will crash horribly when multiple
    // threads are involved.
    DenseVertexMap<float> heightDiff;
    heightDiff.reserve(mesh.nextVertexIndex());

    // Output
    string msg = timestamp.getElapsedTime() + "Computing height differences...";
    ProgressBar progress(mesh.numVertices(), msg);
    ++progress;

    std::set<VertexHandle> invalid;

// Calculate height difference for each vertex
#pragma omp parallel for
    for (size_t i = 0; i < mesh.nextVertexIndex(); i++)
    {
        auto vH = VertexHandle(i);
        if (!mesh.containsVertex(vH))
        {
            continue;
        }

        float minHeight = std::numeric_limits<float>::max();
        float maxHeight = std::numeric_limits<float>::lowest();

        visitLocalVertexNeighborhood(mesh, invalid, vH, radius, [&](auto neighbor) {
            auto curPos = mesh.getVertexPosition(neighbor);

            if (curPos.z < minHeight)
            {
                minHeight = curPos.z;
            }
            if (curPos.z > maxHeight)
            {
                maxHeight = curPos.z;
            }
        });

// Calculate the final height difference
#pragma omp critical
        {
            heightDiff.insert(vH, maxHeight - minHeight);
            ++progress;
        }
    }

    if(!timestamp.isQuiet())
    cout << endl;

  if (!invalid.empty())
    {
        std::cerr << "Found " << invalid.size() << " invalid, non manifold "
            << "vertices." << std::endl;
    }

    return heightDiff;
}

template <typename BaseVecT>
DenseEdgeMap<float> calcVertexAngleEdges(const BaseMesh<BaseVecT> &mesh, const VertexMap<Normal<typename BaseVecT::CoordType>> &normals)
{
    DenseEdgeMap<float> edgeAngle(mesh.nextEdgeIndex(), 0);
    for (auto eH : mesh.edges())
    {
        auto vHVector = mesh.getVerticesOfEdge(eH);
        edgeAngle.insert(eH, acos(normals[vHVector[0]].dot(normals[vHVector[1]])));
        if (isnan(edgeAngle[eH]))
            edgeAngle[eH] = 0;
    }
    return edgeAngle;
}

template <typename BaseVecT>
DenseVertexMap<float> calcAverageVertexAngles(
    const BaseMesh<BaseVecT> &mesh,
    const VertexMap<Normal<typename BaseVecT::CoordType>> &normals)
{
    DenseVertexMap<float> vertexAngles(mesh.nextVertexIndex(), 0);
    auto edgeAngles = calcVertexAngleEdges(mesh, normals);
    std::set<VertexHandle> invalid;

    for (auto vH : mesh.vertices())
    {
        float angleSum = 0;
        try
        {
            auto edgeVec = mesh.getEdgesOfVertex(vH);
            int degree = edgeVec.size();
            for (auto eH : edgeVec)
            {
                angleSum += edgeAngles[eH];
            }
            vertexAngles.insert(vH, angleSum / degree);
        }
        catch (lvr2::PanicException exception)
        {
            vertexAngles.insert(vH, M_PI);
            invalid.insert(vH);
        }
        catch (VertexLoopException exception)
        {
            vertexAngles.insert(vH, M_PI);
            invalid.insert(vH);
        }
    }
    if (!invalid.empty())
    {
        std::cerr << std::endl << "Found " << invalid.size()
            << " invalid, non manifold vertices." << std::endl
            << "The average vertex angle of the invalid vertices has been set"
            << " to Pi." << std::endl;
    }
    return vertexAngles;
}

template <typename BaseVecT>
DenseVertexMap<float> calcVertexRoughness(
    const BaseMesh<BaseVecT> &mesh,
    double radius,
    const VertexMap<Normal<typename BaseVecT::CoordType>> &normals)
{
    // We create a map to store the roughness for each vertex. We preallocate
    // memory for all vertices. This is not only an optimization, but more
    // importantly to avoid multithreading crashes. The parallelized loop
    // further down inserts into this data structure; if there is no free
    // memory left, it will reallocate which will crash horribly when multiple
    // threads are involved.
    DenseVertexMap<float> roughness;
    roughness.reserve(mesh.nextVertexIndex());

    auto averageAngles = calcAverageVertexAngles(mesh, normals);

    // Output
    string msg = timestamp.getElapsedTime() + "Computing roughness";
    ProgressBar progress(mesh.numVertices(), msg);
    ++progress;

    std::set<VertexHandle> invalid;

// Calculate roughness for each vertex
#pragma omp parallel for
    for (size_t i = 0; i < mesh.nextVertexIndex(); i++)
    {
        auto vH = VertexHandle(i);
        if (!mesh.containsVertex(vH))
        {
            continue;
        }

        float sum = 0.0;
        size_t count = 0;

        visitLocalVertexNeighborhood(mesh, invalid, vH, radius, [&](auto neighbor) {
            sum += averageAngles[neighbor];
            count += 1;
        });

#pragma omp critical
        {
            // Calculate the final roughness
            roughness.insert(vH, count ? sum / count : 0);
            ++progress;
        }
    }
    if(!timestamp.isQuiet())
        cout << endl;

    if (!invalid.empty())
    {
        std::cerr << "Found " << invalid.size() << " invalid, non manifold "
            << "vertices." << std::endl;
    }
    
    return roughness;
}

template <typename BaseVecT>
void calcVertexRoughnessAndHeightDifferences(
    const BaseMesh<BaseVecT> &mesh,
    double radius,
    const VertexMap<Normal<typename BaseVecT::CoordType>> &normals,
    DenseVertexMap<float> &roughness,
    DenseVertexMap<float> &heightDiff)
{
    // Reserving memory in those maps is important to avoid multi threading
    // related crashes.
    roughness.clear();
    roughness.reserve(mesh.nextVertexIndex());
    heightDiff.clear();
    heightDiff.reserve(mesh.nextVertexIndex());

    std::set<VertexHandle> invalid;
    auto averageAngles = calcAverageVertexAngles(mesh, normals);

// Calculate roughness and height difference for each vertex
#pragma omp parallel for
    for (size_t i = 0; i < mesh.nextVertexIndex(); i++)
    {
        auto vH = VertexHandle(i);
        if (!mesh.containsVertex(vH))
        {
            continue;
        }

        double sum = 0.0;
        uint32_t count = 0;
        float minHeight = std::numeric_limits<float>::max();
        float maxHeight = std::numeric_limits<float>::lowest();

        visitLocalVertexNeighborhood(mesh, invalid, vH, radius, [&](auto neighbor) {
            sum += averageAngles[neighbor];
            count += 1;

            auto curPos = mesh.getVertexPosition(neighbor);
            if (curPos.z < minHeight)
            {
                minHeight = curPos.z;
            }
            if (curPos.z > maxHeight)
            {
                maxHeight = curPos.z;
            }
        });

#pragma omp critical
        {
            // Calculate the final roughness
            roughness.insert(vH, count ? sum / count : 0);

            // Calculate the final height difference
            heightDiff.insert(vH, maxHeight - minHeight);
        }
    }
    if (!invalid.empty())
    {
        std::cerr << "Found " << invalid.size() << " invalid, non manifold "
            << "vertices." << std::endl;
    }
}

template <typename BaseVecT>
DenseEdgeMap<float> calcVertexDistances(const BaseMesh<BaseVecT> &mesh)
{
    DenseEdgeMap<float> distances;

    distances.clear();
    distances.reserve(mesh.nextEdgeIndex());
    for (auto eH : mesh.edges())
    {
        auto vertices = mesh.getVerticesOfEdge(eH);
        const float dist = mesh.getVertexPosition(vertices[0]).distance(mesh.getVertexPosition(vertices[1]));
        distances.insert(eH, dist);
    }
    return distances;
}

class CompareDist
{
public:
    bool operator()(pair<lvr2::VertexHandle, float> n1, pair<lvr2::VertexHandle, float> n2)
    {
        return n1.second > n2.second;
    }
};

template <typename BaseVecT>
bool Dijkstra(
    const BaseMesh<BaseVecT> &mesh,
    const VertexHandle &start,
    const VertexHandle &goal,
    const DenseEdgeMap<float> &edgeCosts,
    std::list<VertexHandle> &path,
    DenseVertexMap<float> &distances,
    DenseVertexMap<VertexHandle> &predecessors,
    DenseVertexMap<bool> &seen,
    DenseVertexMap<float> &vertex_costs)
{
    path.clear();
    distances.clear();
    predecessors.clear();

    // initialize distances with infinity
    // initialize predecessor of each vertex with itself
    for (auto const &vH : mesh.vertices())
    {
        distances.insert(vH, std::numeric_limits<float>::infinity());
        predecessors.insert(vH, vH);
    }

    distances[start] = 0;

    if (goal == start)
    {
        return true;
    }

    std::priority_queue<pair<VertexHandle, float>, vector<pair<VertexHandle, float>>, CompareDist> pq;

    pair<VertexHandle, float> first_pair(start, 0);
    pq.push(first_pair);

    while (!pq.empty())
    {
        pair<VertexHandle, float> pair = pq.top();
        VertexHandle current_vh = pair.first;
        float current_dist = pair.second;
        pq.pop();

        // Check if the current Vertex was seen already
        if (seen[current_vh])
        {
            // Skip to while
            continue;
        }
        // Set the seen vertex to True
        seen[current_vh] = true;

        // Get all edges from the current Vertex
        std::vector<VertexHandle> neighbours;
        mesh.getNeighboursOfVertex(current_vh, neighbours);

        for (auto neighbour_vh : neighbours)
        {
            if (seen[neighbour_vh] || vertex_costs[neighbour_vh] >= 1)
                continue;

            float edge_cost = edgeCosts[mesh.getEdgeBetween(current_vh, neighbour_vh).unwrap()];

            float tmp_neighbour_cost = distances[current_vh] + edge_cost;
            if (distances[neighbour_vh] > tmp_neighbour_cost)
            {
                distances[neighbour_vh] = tmp_neighbour_cost;
                predecessors[neighbour_vh] = current_vh;
                pq.push(std::pair<VertexHandle, float>(neighbour_vh, tmp_neighbour_cost));
            }
        }
    }

    VertexHandle prev = goal;

    if (prev == predecessors[goal])
        return false;

    do
    {
        path.push_front(prev);
        prev = predecessors[prev];
    } while (prev != start);

    path.push_front(start);

    return true;
}

} // namespace lvr2
