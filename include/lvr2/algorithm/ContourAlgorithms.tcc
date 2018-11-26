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
 * ContourAlgorithms.tcc
 */

#include <vector>
#include <algorithm>

using std::vector;

namespace lvr2
{

template<typename BaseVecT, typename VisitorF>
void walkContour(
    const BaseMesh<BaseVecT>& mesh,
    EdgeHandle startH,
    VisitorF visitor
)
{
    walkContour(mesh, startH, visitor, [](auto unused)
    {
        return true;
    });
}

template<typename BaseVecT, typename VisitorF, typename PredF>
void walkContour(const BaseMesh<BaseVecT>& mesh, EdgeHandle startH, VisitorF visitor, PredF exists)
{
    const auto faces = mesh.getFacesOfEdge(startH);
    const auto numFaces = (faces[0] && exists(faces[0].unwrap()) ? 1 : 0)
        + (faces[1] && exists(faces[1].unwrap()) ? 1 : 0);

    if (numFaces == 0)
    {
        panic("attempt to walk a contour starting at a lonely edge!");
    }
    if (numFaces == 2)
    {
        panic("attempt to walk a contour starting at a non-boundary edge!");
    }

    // This is only used later, but created here to avoid unecessary heap
    // allocations.
    vector<EdgeHandle> edgesOfVertex;

    // The one face our edge is adjacent to. We treat faces for which
    // `exists()` returns false as non-existent.
    const auto startFace = faces[0] && exists(faces[0].unwrap())
        ? faces[0].unwrap()
        : faces[1].unwrap();
    const auto startVertices = mesh.getVerticesOfEdge(startH);

    // First we need to find the correct next edge in the contour. The problem
    // is that `getVerticesOfEdge()` returns the vertices in unknown order. We
    // can't know which vertex is the one "in counter-clockwise" direction,
    // which we need to know.
    //
    // Luckily, we can find out by using the fact that `startH` mustn't be a
    // lonely edge (it has exactly one face) and that we can get all vertices
    // of a face in counter-clockwise order.
    //
    // This means that our correct start-vertex is the one coming after the
    // other start vertex in the face's vertices.
    const auto vertices = mesh.getVerticesOfFace(startFace);
    const auto firstIt = std::find(vertices.begin(), vertices.end(), startVertices[0]);
    const auto secondIt = std::find(vertices.begin(), vertices.end(), startVertices[1]);

    // We simply can find out which vertex is coming "after" the other one by
    // looking at the difference in their indices in the array.
    //
    // Case: `second` comes after `first`
    // ----------------------------------
    // ┌───┬───┬───┐        ┌───┬───┬───┐        ┌───┬───┬───┐
    // │ F │ S │   │   or   │   │ F │ S │   or   │ S │   │ F │
    // └───┴───┴───┘        └───┴───┴───┘        └───┴───┴───┘
    //   (diff 1)             (diff 1)             (diff -2)
    //
    //
    // Case: `first` comes after `second`
    // ----------------------------------
    // ┌───┬───┬───┐        ┌───┬───┬───┐        ┌───┬───┬───┐
    // │ S │ F │   │   or   │   │ S │ F │   or   │ F │   │ S │
    // └───┴───┴───┘        └───┴───┴───┘        └───┴───┴───┘
    //   (diff -1)            (diff -1)            (diff 2)
    //
    //
    const auto diff = secondIt - firstIt;

    // This stores the vertex of the last edge we visited which connects said
    // edge with the `currEdge` in counter-clockwise order.
    auto currVertexH = (diff == 1 || diff == -2) ? *secondIt : *firstIt;

    // This holds the edge we are currently looking at.
    auto currEdgeH = startH;

    do
    {
        // Call the visitor
        visitor(currVertexH, currEdgeH);

        // Determine the next vertex. Since we know the last one, we can simply
        // find out which one is the correct one.
        const auto vertices = mesh.getVerticesOfEdge(currEdgeH);
        const auto nextVertexH = vertices[0] == currVertexH ? vertices[1] : vertices[0];

        // Determine next edge. It's the first edge after our current one (in
        // the clockwise ordered list of edges of the next vertex) which
        // touches at least one existing face.
        edgesOfVertex.clear();
        mesh.getEdgesOfVertex(nextVertexH, edgesOfVertex);

        const auto ourPos = std::find(edgesOfVertex.begin(), edgesOfVertex.end(), currEdgeH) - edgesOfVertex.begin();
        auto afterPos = ourPos;
        while(true)
        {
            afterPos = (afterPos + 1) % edgesOfVertex.size();
            if(afterPos == ourPos)
            {
                // This means we looped around the vertex once without finding
                // a fitting edge. But this can't be since there mustn't be any
                // lonely edges. So the previous edge is adjacent to exactly
                // one existing face. This obviously implies that there is
                // another such edge around the vertex, which we would have
                // found if `exists()` wouldn't give us inconsistent results!
                panic(
                    "Unable to find next edge. This means that the given `exists` "
                    "argument is not consistent. Please fix that"
                );
            }

            const auto facesOfEdge = mesh.getFacesOfEdge(edgesOfVertex[afterPos]);
            const auto touchesAnExistingFace =
                (facesOfEdge[0] && exists(facesOfEdge[0].unwrap())) ||
                (facesOfEdge[1] && exists(facesOfEdge[1].unwrap()));

            if (touchesAnExistingFace)
            {
                break;
            }
        }

        // Assign values for the next iteration
        currEdgeH = edgesOfVertex[afterPos];
        currVertexH = nextVertexH;
    } while (currEdgeH != startH);
}

template<typename BaseVecT, typename PredF>
void calcContourEdges(
    const BaseMesh<BaseVecT>& mesh,
    EdgeHandle startH,
    vector<EdgeHandle>& contourOut,
    PredF exists
)
{
    walkContour(mesh, startH, [&](auto vertexH, auto edgeH)
    {
        contourOut.push_back(edgeH);
    }, exists);
}

template<typename BaseVecT>
void calcContourEdges(
    const BaseMesh<BaseVecT>& mesh,
    EdgeHandle startH,
    vector<EdgeHandle>& contourOut
)
{
    calcContourEdges(mesh, startH, contourOut, [](auto unused)
    {
        return true;
    });
}

template<typename BaseVecT, typename PredF>
void calcContourVertices(
    const BaseMesh<BaseVecT>& mesh,
    EdgeHandle startH,
    vector<VertexHandle>& contourOut,
    PredF exists
)
{
    walkContour(mesh, startH, [&](auto vertexH, auto edgeH)
    {
        contourOut.push_back(vertexH);
    }, exists);
}

template<typename BaseVecT>
void calcContourVertices(
    const BaseMesh<BaseVecT>& mesh,
    EdgeHandle startH,
    vector<VertexHandle>& contourOut
)
{
    calcContourVertices(mesh, startH, contourOut, [](auto unused)
    {
        return true;
    });
}

} // namespace lvr2
