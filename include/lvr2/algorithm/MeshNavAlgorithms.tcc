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
 * MeshNavAlgorithms.tcc
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
            // only do that if we haven't visited the vertex before.
            auto distSquared = mesh.getVertexPosition(newVH).squaredDistanceFrom(vPos);
            if (!visited[newVH] && distSquared < radiusSquared)
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
    DenseVertexMap<float> heightDiff;

    // Calculate height difference for each vertex
    for (auto vH: mesh.vertices())
    {
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
    DenseVertexMap<float> roughness;

    auto averageAngles = calcAverageVertexAngles(mesh, normals);

    // Calculate roughness for each vertex
    for (auto vH: mesh.vertices())
    {
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
    roughness.clear();
    heightDiff.clear();

    auto averageAngles = calcAverageVertexAngles(mesh, normals);

    // Calculate roughness and height difference for each vertex
    for (auto vH: mesh.vertices())
    {
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
