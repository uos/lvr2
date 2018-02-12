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
 * ReductionAlgorithms.tcc
 */

#include <unordered_set>
#include <vector>

#include <lvr2/algorithm/NormalAlgorithms.hpp>
#include <lvr2/geometry/Handles.hpp>
#include <lvr2/util/Meap.hpp>

using std::unordered_set;
using std::vector;


namespace lvr2
{

struct DirEdge
{
    VertexHandle first;
    VertexHandle second;

    DirEdge(VertexHandle first, VertexHandle second) : first(first), second(second) {}

    bool operator==(const DirEdge& other) const {
        return this->first == other.first && this->second == other.second;
    }
};

} // namespace lvr2



namespace std
{

template <>
class hash<lvr2::DirEdge>
{
public:
    size_t operator()(const lvr2::DirEdge& s) const
    {
        size_t h1 = std::hash<lvr2::VertexHandle>()(s.first);
        size_t h2 = std::hash<lvr2::VertexHandle>()(s.second);
        return h1 ^ h2;
    }
};
}



namespace lvr2
{

template<typename BaseVecT, typename CostF>
size_t iterativeEdgeCollapse(
    BaseMesh<BaseVecT>& mesh,
    const size_t count,
    FaceMap<Normal<BaseVecT>>& faceNormals,
    CostF collapseCost
)
{
    Meap<DirEdge, float> queue(mesh.numEdges() * 2);
    const auto& constFaceNormals = faceNormals;


    // These two variables are only used later, but are created here to avoid
    // unnecessary heap allocations.
    vector<FaceHandle> facesAroundVertex;
    vector<VertexHandle> vertexNeighbors;
    unordered_set<EdgeHandle> upsertEdges;


    // Calculate initial costs of all edges
    for (const auto eH: mesh.edges())
    {
        auto verts = mesh.getVerticesOfEdge(eH);

        for (auto e: {DirEdge(verts[0], verts[1]), DirEdge(verts[1], verts[0])})
        {
            auto maybeCost = collapseCost(e.first, e.second, constFaceNormals);
            if (maybeCost)
            {
                queue.insert(e, *maybeCost);
            }
        }
    }


    size_t collapsedEdgeCount = 0;

    // Repeat `count` times
    while (collapsedEdgeCount < count && !queue.isEmpty())
    {
        // Collapse the edge with minimal cost if it is collapsable.
        const auto min = queue.popMin();
        const auto edgeMin = mesh.getEdgeBetween(min.key.first, min.key.second).unwrap();

        // We can already removed the edge in the opposite direction from the
        // priority queue.
        queue.erase({min.key.second, min.key.first});

        if (!mesh.isCollapsable(edgeMin))
        {
            // If we can't collapse this edge, we will just ignore it.
            continue;
        }

        // Before we can collapse the edge, we need to remove certain directed
        // edges from the priority queue: the ones which share one endpoint
        // with the edge we are about to collapse.
        for (auto centerH: {min.key.first, min.key.second})
        {
            vertexNeighbors.clear();
            mesh.getNeighboursOfVertex(centerH, vertexNeighbors);
            for (auto vH: vertexNeighbors) {
                if (vH != min.key.first && vH != min.key.second) {
                    queue.erase({centerH, vH});
                    queue.erase({vH, centerH});
                }
            }
        }


        auto toPos = mesh.getVertexPosition(min.key.second);
        auto result = mesh.collapseEdge(edgeMin);
        collapsedEdgeCount += 1;

        // Set correct position of the new vertex
        mesh.getVertexPosition(result.midPoint) = toPos;

        facesAroundVertex.clear();
        upsertEdges.clear();

        // Remove all entries from that map that belong to now invalid handles
        // and add values for the handles that were created.
        for (auto neighbor: result.neighbors)
        {
            if (neighbor)
            {
                faceNormals.erase(neighbor->removedFace);
            }
        }

        // We collect all faces around the new vertex and insert all edges of
        // those faces into a set, to get a unique list of edges that need to
        // be updated. We also update the normal of all those faces.
        mesh.getFacesOfVertex(result.midPoint, facesAroundVertex);
        for (auto fH: facesAroundVertex)
        {
            auto maybeNormal = getFaceNormal(mesh.getVertexPositionsOfFace(fH));
            auto normal = maybeNormal
                ? *maybeNormal
                : Normal<BaseVecT>(0, 0, 1);

            faceNormals[fH] = normal;
            for (auto eH: mesh.getEdgesOfFace(fH))
            {
                upsertEdges.insert(eH);
            }
        }

        // Update the cost of affected edges
        for (auto eH: upsertEdges)
        {
            auto verts = mesh.getVerticesOfEdge(eH);
            for (auto e: {DirEdge(verts[0], verts[1]), DirEdge(verts[1], verts[0])})
            {
                auto maybeCost = collapseCost(e.first, e.second, constFaceNormals);
                if (maybeCost)
                {
                    queue.insert(e, *maybeCost);
                }
                else
                {
                    queue.erase(e);
                }
            }
        }
    }

    return collapsedEdgeCount;
}

template<typename BaseVecT>
optional<float> collapseCostSimpleNormalDiff(
    const BaseMesh<BaseVecT>& mesh,
    const FaceMap<Normal<BaseVecT>>& normals,
    VertexHandle fromH,
    VertexHandle toH
)
{
    // Get the edge handle and the 0--2 adjacent faces
    auto eH = mesh.getEdgeBetween(fromH, toH).unwrap();
    auto adjacentFaces = mesh.getFacesOfEdge(eH);

    // If the edge is lonely, we won't collapse it
    if (!adjacentFaces[0] || !adjacentFaces[1])
    {
        return boost::none;
    }

    // Calculate the curvature term
    float curvature = 0.0;
    auto facesAroundFrom = mesh.getFacesOfVertex(fromH);
    if (facesAroundFrom.size() != mesh.getEdgesOfVertex(fromH).size())
    {
        return boost::none;
    }

    for (auto fH: facesAroundFrom)
    {
        // The diff is between 0 and 1, so 2 is a valid start value to find
        // the minimum.
        double minDiff = 2.0;
        if (adjacentFaces[0])
        {
            auto dot = normals[adjacentFaces[0].unwrap()].dot(normals[fH].asVector());
            // We are the first, so we know our value is smaller than the
            // initial one
            minDiff = (1.0 - dot) / 2.0;
        }
        if (adjacentFaces[1])
        {
            auto dot = normals[adjacentFaces[1].unwrap()].dot(normals[fH].asVector());
            auto diff = (1.0 - dot) / 2.0;
            if (diff < minDiff)
            {
                minDiff = diff;
            }
        }

        // Find the maximum
        if (minDiff > curvature)
        {
            curvature = minDiff;
        }
    }

    // Calculate length
    auto length = mesh.getVertexPosition(fromH).distanceFrom(mesh.getVertexPosition(toH));

    return length * curvature;
}


} // namespace lvr2
