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
 * CleanupAlgorithms.tcc
 */

#include "lvr2/io/Progress.hpp"
#include "lvr2/algorithm/ContourAlgorithms.hpp"
#include "lvr2/attrmaps/AttrMaps.hpp"
#include "lvr2/io/Timestamp.hpp"

namespace lvr2
{

template<typename BaseVecT>
void cleanContours(BaseMesh<BaseVecT>& mesh, int iterations, float areaThreshold)
{
    for (int i = 0; i < iterations; i++)
    {
        for (const auto fH: mesh.faces())
        {
            // For each face, we want to count the number of boundary edges
            // adjacent to that face. This can be a number between 0 and 3.
            int boundaryEdgeCount = 0;
            for (const auto eH: mesh.getEdgesOfFace(fH))
            {
                // For both (optional) faces of the edge...
                for (const auto neighborFaceH: mesh.getFacesOfEdge(eH))
                {
                    // ... we will count one up if there is no face on that
                    // side. Note that this correctly ignores our own face.
                    if (!neighborFaceH)
                    {
                        boundaryEdgeCount += 1;
                    }
                }
            }

            // Now, given the number of boundary edges, we decide what to do
            // with the face.
            if (boundaryEdgeCount >= 2)
            {
                mesh.removeFace(fH);
            }
            else if (boundaryEdgeCount == 1 && mesh.calcFaceArea(fH) < areaThreshold)
            {
                mesh.removeFace(fH);
            }
        }
    }
}


template<typename BaseVecT>
size_t naiveFillSmallHoles(BaseMesh<BaseVecT>& mesh, size_t maxSize, bool collapseOnly)
{
    // There are no holes with less than three edges.
    if (maxSize < 3)
    {
        return 0;
    }

    cout << timestamp << "Trying to remove all holes with size ≤ " << maxSize << endl;


    // First, we need to have a ClusterBiMap where each cluster describes one
    // connected part of the mesh.
    auto subMeshes = clusterGrowing(mesh, [](FaceHandle referenceFaceH, FaceHandle currentFaceH)
    {
        return true;
    });

    DenseEdgeMap<bool> visitedEdges(mesh.numEdges(), false);

    // We count how many holes we were unable to fill. This happens when a hole
    // has many non-collapsable edges.
    size_t failedToFillCount = 0;

    // This is only used later but created here to avoid useless heap
    // allocations.
    vector<vector<EdgeHandle>> contours;

    // We execute the algorithm for each connected part of the mesh
    string comment = timestamp.getElapsedTime() + "Trying to remove all holes ";

    ProgressBar progress(subMeshes.numCluster(), comment);
    for (auto clusterH: subMeshes)
    {
        if(!timestamp.isQuiet())
            ++progress;
        contours.clear();

        // We only use this within the loop, but create it here to avoid
        // useless heap allocations.
        vector<EdgeHandle> contourEdges;

        for (auto faceH: subMeshes[clusterH])
        {
            for (auto eH: mesh.getEdgesOfFace(faceH))
            {
                // Skip already visited edges and mark edge as visited
                if (visitedEdges[eH])
                {
                    continue;
                }
                visitedEdges[eH] = true;

                // We are only interested in boundary edges
                if (mesh.numAdjacentFaces(eH) != 1)
                {
                    continue;
                }

                // Get full contour and store it in our list
                contourEdges.clear();
                calcContourEdges(mesh, eH, contourEdges);
                contours.emplace_back(contourEdges);

                // Mark all edges in the contour we just added as visited
                for (auto edgeH: contours.back())
                {
                    visitedEdges[edgeH] = true;
                }
            }
        }

        // It can actually happen that a cluster doesn't contain any contour
        // edges that we haven't visited before. In that case, just ignore it.
        if (contours.size() == 0)
        {
            continue;
        }

        // Now we have a list of all contours of the current part of the mesh.
        // Next, we need to find the contour with most vertices and delete it
        // from the list. This is a very naive assumption: we say that the
        // contour containing most edges is the "outer" contour which we don't
        // treat as a hole. This is far from being universally correct, but
        // works OK for small holes.
        size_t maxIdx = 0;
        for (size_t i = 1; i < contours.size(); i++)
        {
            if (contours[i].size() > contours[maxIdx].size())
            {
                maxIdx = i;
            }
        }
        contours.erase(contours.begin() + maxIdx);

        // We assume that all remaining contours are holes we could fill. (And
        // yes, we need the index later and can't use a range based for loop).
        for (size_t contourIdx = 0; contourIdx < contours.size(); contourIdx++)
        {
            auto& contour = contours[contourIdx];

            // We only try to fill the hole if it is smaller than the threshold
            if (contour.size() > maxSize)
            {
                continue;
            }

            // Collapse as many edges as possible, but already stop when we
            // have less than 4 edges left.
            while (contour.size() > 3)
            {
                auto collapsableEdge = std::find_if(contour.begin(), contour.end(), [&](auto edgeH)
                {
                    return mesh.isCollapsable(edgeH);
                });

                // In case there is no collapsable edge anymore
                if (collapsableEdge == contour.end())
                {
                    break;
                }

                // Collapse edge and remove it from the list of contours
                auto collapseResult = mesh.collapseEdge(*collapsableEdge);
                contour.erase(collapsableEdge);

                // It may happen that the edge collapse invalidated an edge
                // from our list. We need to fix that.
                //
                // First, we know that only one face will be collapsed, so we
                // can just deal with that one.
                auto removedNeighbor = collapseResult.neighbors[0]
                    ? *(collapseResult.neighbors[0])
                    : *(collapseResult.neighbors[1]);

                // We have to check if the removed edges are referenced
                // anywhere. If yes, we have to fix those occurrences to avoid
                // referencing invalid edges later. But we only store boundary
                // edges, so we only need to fix anything if the new edge is
                // still a boundary edge. That means that it is referred to
                // somewhere.
                if (mesh.numAdjacentFaces(removedNeighbor.newEdge) < 2)
                {
                    for (auto removedEdgeH: removedNeighbor.removedEdges)
                    {
                        for (size_t i = contourIdx; i < contours.size(); i++)
                        {
                            auto& contourToFix = contours[i];
                            auto it = std::find(contourToFix.begin(), contourToFix.end(), removedEdgeH);
                            if (it != contourToFix.end())
                            {
                                // We replace the handle with the new one
                                *it = removedNeighbor.newEdge;

                                // We can stop here, because each edge is only
                                // part of one contour.
                                break;
                            }
                        }
                    }
                }

                // If the removed face belongs to a cluster different from the
                // current one, we also have to remove the face there.
                auto clusterH = subMeshes.getClusterOf(removedNeighbor.removedFace);
                if (clusterH)
                {
                    subMeshes.removeFromCluster(clusterH.unwrap(), removedNeighbor.removedFace);
                }
            }

            if (collapseOnly)
            {
                continue;
            }

            // We collapsed as many edges as we could. In most cases, there is
            // only one or two triangles left which we can close right ahead.
            // Otherwise, we just leave the hole.
            if (contour.size() == 3)
            {
                // The edges in `contour` are already in the correct order.
                auto v0 = mesh.getVertexBetween(contour[0], contour[1]).unwrap();
                auto v1 = mesh.getVertexBetween(contour[1], contour[2]).unwrap();
                auto v2 = mesh.getVertexBetween(contour[2], contour[0]).unwrap();
                mesh.addFace(v0, v1, v2);
            }
            else
            {
                failedToFillCount += 1;
            }
        }
    }

    if(!timestamp.isQuiet())
        cout << endl;

    return failedToFillCount;
}


} // namespace lvr2
