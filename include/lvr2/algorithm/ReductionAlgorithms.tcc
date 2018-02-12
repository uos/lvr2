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

template<typename BaseVecT, typename CostF>
size_t iterativeEdgeCollapse(
    BaseMesh<BaseVecT>& mesh,
    const size_t count,
    FaceMap<Normal<BaseVecT>>& faceNormals,
    CostF collapseCost
)
{
    Meap<EdgeHandle, float, DenseAttrMap> queue(mesh.numEdges());
    const auto& constFaceNormals = faceNormals;

    // Calculate initial costs of all edges
    for (const auto eH: mesh.edges())
    {
        auto maybeCost = collapseCost(eH, constFaceNormals);
        if (maybeCost)
        {
            queue.insert(eH, *maybeCost);
        }
    }

    // These two variables are only used later, but are created here to avoid
    // unnecessary heap allocations.
    vector<FaceHandle> facesAroundVertex;
    unordered_set<EdgeHandle> affectedEdges;

    size_t collapsedEdgeCount = 0;

    // Repeat `count` times
    while (collapsedEdgeCount < count && !queue.isEmpty())
    {
        // Collapse the edge with minimal cost if it is collapsable.
        const auto min = queue.popMin();
        if (!mesh.isCollapsable(min.key))
        {
            // If we can't collapse this edge, we will just ignore it.
            continue;
        }

        auto result = mesh.collapseEdge(min.key);
        collapsedEdgeCount += 1;

        facesAroundVertex.clear();
        affectedEdges.clear();

        // Remove all entries from that map that belong to now invalid handles
        // and add values for the handles that were created.
        for (auto neighbor: result.neighbors)
        {
            if (neighbor)
            {
                faceNormals.erase(neighbor->removedFace);
                queue.erase(neighbor->removedEdges[0]);
                queue.erase(neighbor->removedEdges[1]);
                affectedEdges.insert(neighbor->newEdge);
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
                affectedEdges.insert(eH);
            }
        }

        // Update the cost of affected edges
        for (auto eH: affectedEdges)
        {
            if (queue.containsKey(eH))
            {
                auto maybeCost = collapseCost(eH, constFaceNormals);
                if (maybeCost)
                {
                    queue.insert(eH, *maybeCost);
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
    EdgeHandle eH
)
{
    const auto vertices = mesh.getVerticesOfEdge(eH);

    unordered_set<FaceHandle> faces;
    for (auto vH: vertices)
    {
        for (auto fH: mesh.getFacesOfVertex(vH))
        {
            faces.insert(fH);
        }
    }

    // In this case we are dealing with a super lonely edge: only two vertices,
    // but no faces adjacent.
    if (faces.empty())
    {
        return boost::none;
    }

    vector<Normal<BaseVecT>> faceNormals;
    for (auto fH: faces)
    {
        faceNormals.push_back(normals[fH]);
    }

    auto avgNormal = Normal<BaseVecT>::average(faceNormals);

    auto sumAngleCost = 0.0;
    for (auto normal: faceNormals)
    {
        sumAngleCost += 1 - normal.dot(avgNormal.asVector());
    }
    auto angleCost = sumAngleCost / faces.size();

    // A long edge is worse than a short one...
    auto lengthCost = mesh.getVertexPosition(vertices[0]).distanceFrom(mesh.getVertexPosition(vertices[1]));



    // Return weighted costs
    return 1.0 * angleCost
        + 0.01 * lengthCost;
}


} // namespace lvr2
