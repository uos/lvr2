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
 * ReductionAlgorithms.tcc
 */

#include <unordered_set>
#include <vector>

#include "lvr2/io/Progress.hpp"
#include "lvr2/algorithm/NormalAlgorithms.hpp"
#include "lvr2/geometry/Handles.hpp"
#include "lvr2/util/Meap.hpp"

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
    FaceMap<Normal<typename BaseVecT::CoordType>>& faceNormals,
    CostF collapseCost
)
{

    std::cout << timestamp << "Reduce mesh by collapsing " << count << " edges" << std::endl;

    Meap<VertexHandle, float> queue(mesh.nextVertexIndex());
    DenseVertexMap<VertexHandle> bestEdge;
    bestEdge.reserve(mesh.nextVertexIndex());

    const auto& constFaceNormals = faceNormals;


    // These two variables are only used later, but are created here to avoid
    // unnecessary heap allocations.
    vector<FaceHandle> facesAroundMidpoint;
    vector<VertexHandle> midpointNeighbors;

    // We need to update the values of a given vertex at several points in
    // this function, so we write a small lambda to do it for us.
    vector<VertexHandle> vertexUpdateNeighbors;
    auto updateVertex = [&](VertexHandle fromH)
    {
        vertexUpdateNeighbors.clear();
        mesh.getNeighboursOfVertex(fromH, vertexUpdateNeighbors);

        // We are trying to find the outgoing edge with the best score. The
        // second vertex is the vertex itself if no outgoing edge is
        // collapsable.
        auto bestToH = fromH;
        auto bestCost = std::numeric_limits<float>::max();

        for (const auto toH: vertexUpdateNeighbors)
        {
            auto maybeCost = collapseCost(fromH, toH, constFaceNormals);
            if (maybeCost)
            {
                if (*maybeCost < bestCost)
                {
                    bestCost = *maybeCost;
                    bestToH = toH;
                }
            }
        }

        if (bestToH != fromH)
        {
            queue.insert(fromH, bestCost);
            bestEdge.insert(fromH, bestToH);
        }
        else
        {
            queue.erase(fromH);
        }
    };

    // Output
    string msg_init = timestamp.getElapsedTime()
        + "Computing all costs for all edges ";
    ProgressBar progress_init(mesh.numVertices() + 1, msg_init);
    ++progress_init;

    // Calculate initial costs of all edges
    for (const auto fromH: mesh.vertices())
    {
        updateVertex(fromH);
        ++progress_init;
    }

    // Output
    string msg = timestamp.getElapsedTime()
        + "Collapsing up to "
        + std::to_string(count)
        + "of the edges ";
    ProgressBar progress(count + 1, msg);
    ++progress;

    size_t collapsedEdgeCount = 0;

    // Repeat `count` times
    while (collapsedEdgeCount < count && !queue.isEmpty())
    {
        // Collapse the edge with minimal cost if it is collapsable.
        const auto fromH = queue.popMin().key();
        const auto toH = bestEdge[fromH];
        const auto edgeMin = mesh.getEdgeBetween(fromH, toH).unwrap();

        if (!mesh.isCollapsable(edgeMin))
        {
            // If we can't collapse this edge, we will just ignore it.
            continue;
        }

        ++progress;


        auto toPos = mesh.getVertexPosition(toH);
        auto result = mesh.collapseEdge(edgeMin);
        collapsedEdgeCount += 1;

        // Set correct position of the new vertex
        mesh.getVertexPosition(result.midPoint) = toPos;

        // If the `to` vertex was really removed, we have to remove it from
        // the queue.
        if (result.midPoint != toH)
        {
            queue.erase(toH);
        }

        facesAroundMidpoint.clear();
        midpointNeighbors.clear();

        // Now we just need to update the best edge for the midpoint and all
        // its neighbors.
        updateVertex(result.midPoint);
        mesh.getNeighboursOfVertex(result.midPoint, midpointNeighbors);
        for (const auto vH: midpointNeighbors)
        {
            updateVertex(vH);
        }

        // We update the normal of all faces touching the midpoint.
        mesh.getFacesOfVertex(result.midPoint, facesAroundMidpoint);
        for (auto fH: facesAroundMidpoint)
        {
            auto maybeNormal = getFaceNormal(mesh.getVertexPositionsOfFace(fH));
            auto normal = maybeNormal
                ? *maybeNormal
                : Normal<typename BaseVecT::CoordType>(0, 0, 1);

            faceNormals[fH] = normal;
        }

        // Remove all entries from that map that belong to now invalid handles
        // and add values for the handles that were created.
        for (auto neighbor: result.neighbors)
        {
            if (neighbor)
            {
                faceNormals.erase(neighbor->removedFace);
            }
        }
    }


    cout << endl << timestamp << "Collapsed " << collapsedEdgeCount << " edges..." << endl;

    return collapsedEdgeCount;
}

template<typename BaseVecT>
size_t simpleMeshReduction(
    BaseMesh<BaseVecT>& mesh,
    const size_t count,
    FaceMap<Normal<typename BaseVecT::CoordType>>& faceNormals
)
{
    vector<EdgeHandle> edgesAroundFrom;
    vector<FaceHandle> facesAroundFrom;

    return iterativeEdgeCollapse(mesh, count, faceNormals, [&](
        VertexHandle fromH,
        VertexHandle toH,
        const FaceMap<Normal<typename BaseVecT::CoordType>>& normals
    ) -> boost::optional<float>
    {
        // The minimal value of the dot product between two normals that is allowed.
        const float MIN_NORMAL_DIFF = 0.5;


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

        facesAroundFrom.clear();
        edgesAroundFrom.clear();
        mesh.getFacesOfVertex(fromH, facesAroundFrom);
        mesh.getEdgesOfVertex(fromH, edgesAroundFrom);
        if (facesAroundFrom.size() != edgesAroundFrom.size())
        {
            return boost::none;
        }

        for (auto fH: facesAroundFrom)
        {
            // Get the two other vertices of this face.
            auto verts = mesh.getVerticesOfFace(fH);
            auto fromPos = verts[0] == fromH
                ? 0
                : (verts[1] == fromH ? 1 : 2);

            auto v1H = verts[(fromPos + 1) % 3];
            auto v2H = verts[(fromPos + 2) % 3];

            // Check if we are looking at one face connected to both `to` and
            // `from`. We can and need to ignore those faces.
            if (v1H == toH || v2H == toH)
            {
                continue;
            }

            std::array<BaseVecT, 3> f_verts = {
                mesh.getVertexPosition(toH),
                mesh.getVertexPosition(v1H),
                mesh.getVertexPosition(v2H)
            };

            // We calculate the normal that the face would have if the edge in
            // question would be collapsed.
            boost::optional<Normal<typename BaseVecT::CoordType>> newNormal = getFaceNormal(f_verts);

            // If the face will have 0 area, we don't want to collapse this edge
            if (!newNormal)
            {
                return boost::none;
            }

            // If the new normal is too different from the old one, we don't want
            // to collapse this edge.
            auto oldNormal = normals[fH];
            if (newNormal->dot(oldNormal) < MIN_NORMAL_DIFF)
            {
                return boost::none;
            }


            // The diff is between 0 and 1, so 2 is a valid start value to find
            // the minimum.
            double minDiff = 2.0;
            if (adjacentFaces[0])
            {
                auto dot = normals[adjacentFaces[0].unwrap()].dot(normals[fH]);
                // We are the first, so we know our value is smaller than the
                // initial one
                minDiff = (1.0 - dot) / 2.0;
            }
            if (adjacentFaces[1])
            {
                auto dot = normals[adjacentFaces[1].unwrap()].dot(normals[fH]);
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
    });
}

} // namespace lvr2
