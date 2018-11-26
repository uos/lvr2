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

#include <lvr2/attrmaps/AttrMaps.hpp>


namespace lvr2
{

template <typename BaseVecT>
void calcVertexLocalNeighborhood(
    const BaseMesh<BaseVecT>& mesh,
    VertexHandle vH,
    double radius,
    vector<VertexHandle>& neighborsOut
)
{
    visitLocalNeighborhoodOfVertex(mesh, vH, radius, [&](auto newVH) {
        neighborsOut.push_back(newVH);
    });
}

template <typename BaseVecT, typename VisitorF>
void visitLocalNeighborhoodOfVertex(
    const BaseMesh<BaseVecT>& mesh,
    VertexHandle vH,
    double radius,
    VisitorF visitor
)
{
    // Prepare values for the radius test
    auto vPos = mesh.getVertexPosition(vH);
    double radiusSquared = radius * radius;

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
    // TODO: reserve memory once the API allows it
    SparseVertexMap<bool> visited(false);
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
        mesh.getNeighboursOfVertex(curVH, directNeighbors);
        for (auto newVH: directNeighbors)
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
DenseVertexMap<float> calcVertexHeightDiff(const BaseMesh<BaseVecT>& mesh, double radius)
{
    // We create a map to store a height-diff for each vertex. We preallocate
    // memory for all vertices. This is not only an optimization, but more
    // importantly to avoid multithreading crashes. The parallelized loop
    // further down inserts into this data structure; if there is no free
    // memory left, it will reallocate which will crash horribly when multiple
    // threads are involved.
    DenseVertexMap<float> heightDiff;
    heightDiff.reserve(mesh.nextVertexIndex());

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

        visitLocalNeighborhoodOfVertex(mesh, vH, radius, [&](auto neighbor) {
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
        heightDiff.insert(vH, maxHeight - minHeight);
    }

    return heightDiff;
}

template<typename BaseVecT>
DenseEdgeMap<float> calcVertexAngleEdges(const BaseMesh<BaseVecT>& mesh, const VertexMap<Normal<BaseVecT>>& normals)
{
    DenseEdgeMap<float> edgeAngle;

    for (auto eH: mesh.edges())
    {
        auto vHVector = mesh.getVerticesOfEdge(eH);
        edgeAngle.insert(eH, acos(normals[vHVector[0]].dot(normals[vHVector[1]].asVector())));
        if(isnan(edgeAngle[eH]))
        {
                edgeAngle[eH] = 0;
        }
    }
    return edgeAngle;
}

template<typename BaseVecT>
DenseVertexMap<float> calcAverageVertexAngles(
    const BaseMesh<BaseVecT>& mesh,
    const VertexMap<Normal<BaseVecT>>& normals
)
{
    DenseVertexMap<float> vertexAngles;
    auto edgeAngles = calcVertexAngleEdges(mesh, normals);

    for (auto vH: mesh.vertices())
    {
        float angleSum = 0;
        auto edgeVec = mesh.getEdgesOfVertex(vH);
        int degree = edgeVec.size();
        for(auto eH: edgeVec)
        {
            angleSum += edgeAngles[eH];
        }
        vertexAngles.insert(vH, angleSum / degree);
    }
    return vertexAngles;
}


template<typename BaseVecT>
DenseVertexMap<float> calcVertexRoughness(
    const BaseMesh<BaseVecT>& mesh,
    double radius,
    const VertexMap<Normal<BaseVecT>>& normals
)
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

    // Calculate roughness for each vertex
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

        visitLocalNeighborhoodOfVertex(mesh, vH, radius, [&](auto neighbor) {
            sum += averageAngles[neighbor];
            count += 1;
        });

        // Calculate the final roughness
        roughness.insert(vH, sum / count);

    }
    return roughness;

}

template<typename BaseVecT>
void calcVertexRoughnessAndHeightDiff(
    const BaseMesh<BaseVecT>& mesh,
    double radius,
    const VertexMap<Normal<BaseVecT>>& normals,
    DenseVertexMap<float>& roughness,
    DenseVertexMap<float>& heightDiff
)
{
    // Reserving memory in those maps is important to avoid multi threading
    // related crashes.
    roughness.clear();
    roughness.reserve(mesh.nextVertexIndex());
    heightDiff.clear();
    heightDiff.reserve(mesh.nextVertexIndex());

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

        visitLocalNeighborhoodOfVertex(mesh, vH, radius, [&](auto neighbor) {
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

        // Calculate the final roughness
        roughness.insert(vH, sum / count);

        // Calculate the final height difference
        heightDiff.insert(vH, maxHeight - minHeight);
    }
}


} // namespace lvr2
