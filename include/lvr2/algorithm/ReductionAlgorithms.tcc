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

#include <lvr2/geometry/Handles.hpp>
#include <lvr2/util/Meap.hpp>

using std::unordered_set;
using std::vector;


namespace lvr2
{

template<typename BaseVecT, typename CostF>
void iterativeEdgeCollapse(BaseMesh<BaseVecT>& mesh, const size_t count, CostF collapseCost)
{
    Meap<EdgeHandle, float, DenseAttrMap> queue(mesh.numEdges());

    // Calculate initial costs of all edges
    for (const auto eH: mesh.edges())
    {
        queue.insert(eH, collapseCost(eH));
    }

    // These two variables are only used later, but are created here to avoid
    // unnecessary heap allocations.
    vector<FaceHandle> facesAroundVertex;
    unordered_set<EdgeHandle> affectedEdges;

    // Repeat `count` times
    for (size_t i = 0; i < count; i++)
    {
        // If the mesh doesn't contain any edges, we can stop.
        if (queue.isEmpty())
        {
            break;
        }

        // Collapse the edge with minimal cost if it is collapsable.
        const auto min = queue.popMin();
        if (!mesh.isCollapsable(min.key))
        {
            // If we can't collapse this edge, we will just ignore it.
            continue;
        }
        auto result = mesh.collapseEdge(min.key);

        // Remove all entries from that map that belong to now invalid handles
        // and add values for the handles that were created.
        for (auto neighbor: result.neighbors)
        {
            if (neighbor)
            {
                queue.erase(neighbor->removedEdges[0]);
                queue.erase(neighbor->removedEdges[1]);
                queue.insert(neighbor->newEdge, collapseCost(neighbor->newEdge));
            }
        }

        // We collect all faces around the new vertex and insert all edges of
        // those faces into a set, to get a unique list of edges that need to
        // be updated.
        facesAroundVertex.clear();
        affectedEdges.clear();
        mesh.getFacesOfVertex(result.midPoint, facesAroundVertex);
        for (auto fH: facesAroundVertex)
        {
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
                queue.updateValue(eH, collapseCost(eH));
            }
        }
    }
}

template<typename BaseVecT>
float collapseCostSimpleNormalDiff(
    const BaseMesh<BaseVecT>& mesh,
    const FaceMap<Normal<BaseVecT>>& normals,
    EdgeHandle eH
)
{
    auto faces = mesh.getFacesOfEdge(eH);

    // If the edge sits between two faces
    if (faces[0] && faces[1])
    {
        const auto n0 = normals[faces[0].unwrap()];
        const auto n1 = normals[faces[1].unwrap()];
        const auto cosAngle = n0.dot(n1.asVector());

        // If the normals are the same, we return 0, if they point in
        // completely different directions, we return something close to 2.
        return 1 - cosAngle;
    }

    // TODO: this basically says that boundary edges are never collapsed. But
    // this is a stupid metric.
    return 2;
}


} // namespace lvr2
