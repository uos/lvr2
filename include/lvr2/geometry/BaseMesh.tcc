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
 * BaseMesh.tcc
 *
 *  @date 15.06.2017
 *  @author Johan M. von Behren <johan@vonbehren.eu>
 */

#include <algorithm>

#include <cmath>

namespace lvr2
{

template<typename HandleT>
MeshHandleIteratorPtr<HandleT>& MeshHandleIteratorPtr<HandleT>::operator++()
{
    ++(*m_iter);
    return *this;
}

template<typename HandleT>
bool MeshHandleIteratorPtr<HandleT>::operator==(const MeshHandleIteratorPtr<HandleT>& other) const
{
    return *m_iter == *other.m_iter;
}

template<typename HandleT>
bool MeshHandleIteratorPtr<HandleT>::operator!=(const MeshHandleIteratorPtr<HandleT>& other) const
{
    return *m_iter != *other.m_iter;
}

template<typename HandleT>
HandleT MeshHandleIteratorPtr<HandleT>::operator*() const
{
    return **m_iter;
}

template <typename BaseVecT>
FaceIteratorProxy<BaseVecT> BaseMesh<BaseVecT>::faces() const
{
    return FaceIteratorProxy<BaseVecT>(*this);
}

template <typename BaseVecT>
EdgeIteratorProxy<BaseVecT> BaseMesh<BaseVecT>::edges() const
{
    return EdgeIteratorProxy<BaseVecT>(*this);
}

template <typename BaseVecT>
VertexIteratorProxy<BaseVecT> BaseMesh<BaseVecT>::vertices() const
{
    return VertexIteratorProxy<BaseVecT>(*this);
}


template <typename BaseVecT>
MeshHandleIteratorPtr<FaceHandle> FaceIteratorProxy<BaseVecT>::begin() const
{
    return m_mesh.facesBegin();
}

template <typename BaseVecT>
MeshHandleIteratorPtr<FaceHandle> FaceIteratorProxy<BaseVecT>::end() const
{
    return m_mesh.facesEnd();
}

template <typename BaseVecT>
MeshHandleIteratorPtr<EdgeHandle> EdgeIteratorProxy<BaseVecT>::begin() const
{
    return m_mesh.edgesBegin();
}

template <typename BaseVecT>
MeshHandleIteratorPtr<EdgeHandle> EdgeIteratorProxy<BaseVecT>::end() const
{
    return m_mesh.edgesEnd();
}

template <typename BaseVecT>
MeshHandleIteratorPtr<VertexHandle> VertexIteratorProxy<BaseVecT>::begin() const
{
    return m_mesh.verticesBegin();
}

template <typename BaseVecT>
MeshHandleIteratorPtr<VertexHandle> VertexIteratorProxy<BaseVecT>::end() const
{
    return m_mesh.verticesEnd();
}

template <typename BaseVecT>
array<Point<BaseVecT>, 3> BaseMesh<BaseVecT>::getVertexPositionsOfFace(FaceHandle handle) const
{
    auto handles = getVerticesOfFace(handle);

    auto v1 = getVertexPosition(handles[0]);
    auto v2 = getVertexPosition(handles[1]);
    auto v3 = getVertexPosition(handles[2]);

    return {v1, v2, v3};
}

template <typename BaseVecT>
Point<BaseVecT> BaseMesh<BaseVecT>::calcFaceCentroid(FaceHandle handle) const
{
    return Point<BaseVecT>::centroid(
        this->getVertexPositionsOfFace(handle)
    );
}

template <typename BaseVecT>
typename BaseVecT::CoordType BaseMesh<BaseVecT>::calcFaceArea(FaceHandle handle) const
{
    // Calculate the area using Heron's formula. `s` is half the perimeter of
    // the triangle. `a`, `b` and `c` are the length of the three sides of the
    // triangle.
    auto positions = this->getVertexPositionsOfFace(handle);
    auto a = positions[0].distanceFrom(positions[1]);
    auto b = positions[0].distanceFrom(positions[2]);
    auto c = positions[1].distanceFrom(positions[2]);

    auto s = (a + b + c) / 2;

    return sqrt(s * (s - a) * (s - b) * (s - c));
}

template<typename BaseVecT>
template<typename VisitorF>
void BaseMesh<BaseVecT>::walkContour(EdgeHandle startH, VisitorF visitor) const
{
    if (numAdjacentFaces(startH) == 0)
    {
        panic("attempt to walk a contour starting at a lonely edge!");
    }
    if (numAdjacentFaces(startH) == 2)
    {
        panic("attempt to walk a contour starting at a non-boundary edge!");
    }

    // This is only used later, but created here to avoid unecessary heap
    // allocations.
    vector<EdgeHandle> edgesOfVertex;

    // The one face our edge is adjacent to.
    const auto faces = getFacesOfEdge(startH);
    const auto startFace = faces[0] ? faces[0].unwrap() : faces[1].unwrap();
    const auto startVertices = getVerticesOfEdge(startH);

    // First we need to find the correct next edge in the contour. The problem
    // is that `getVerticesOfEdge()` returns the vertices in unknown order. We
    // can't know which vertex is the on "in counter-clockwise" direction,
    // which we need to know.
    //
    // Luckily, we can find out by using the fact that `startH` mustn't be a
    // lonely edge (it has exactly one face) and that we can get all vertices
    // of a face in counter-clockwise order.
    //
    // This means that our correct start-vertex is the one coming after the
    // other start vertex in the face's vertices.
    const auto vertices = getVerticesOfFace(startFace);
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
        const auto vertices = getVerticesOfEdge(currEdgeH);
        const auto nextVertexH = vertices[0] == currVertexH ? vertices[1] : vertices[0];

        // Determine next edge. It's the edge immediately after our current one
        // in the clockwise ordered list of edges of the next vertex.
        edgesOfVertex.clear();
        getEdgesOfVertex(nextVertexH, edgesOfVertex);

        const auto ourPos = std::find(edgesOfVertex.begin(), edgesOfVertex.end(), currEdgeH) - edgesOfVertex.begin();
        const auto afterPos = ourPos == edgesOfVertex.size() - 1 ? 0 : ourPos + 1;

        // Assign values for the next iteration
        currEdgeH = edgesOfVertex[afterPos];
        currVertexH = nextVertexH;
    } while (currEdgeH != startH);
}

template<typename BaseVecT>
void BaseMesh<BaseVecT>::calcContourEdges(EdgeHandle startH, vector<EdgeHandle>& contourOut) const
{
    walkContour(startH, [&](auto vertexH, auto edgeH)
    {
        contourOut.push_back(edgeH);
    });
}

template<typename BaseVecT>
void BaseMesh<BaseVecT>::calcContourVertices(EdgeHandle startH, vector<VertexHandle>& contourOut) const
{
    walkContour(startH, [&](auto vertexH, auto edgeH)
    {
        contourOut.push_back(vertexH);
    });
}

template<typename BaseVecT>
vector<EdgeHandle> BaseMesh<BaseVecT>::calcContourEdges(EdgeHandle startH) const
{
    vector<EdgeHandle> out;
    calcContourEdges(startH, out);
    return out;
}

template<typename BaseVecT>
vector<VertexHandle> BaseMesh<BaseVecT>::calcContourVertices(EdgeHandle startH) const
{
    vector<VertexHandle> out;
    calcContourVertices(startH, out);
    return out;
}

template<typename BaseVecT>
bool BaseMesh<BaseVecT>::isCollapsable(EdgeHandle handle) const
{
    // Collapsing an edge can have a couple of negative side effects:
    //
    // - Creating non-manifold meshes
    // - Changing the Euler-characteristic (changing topology)
    // - Otherwise damaging the mesh
    //
    // Additionally, writing an `edgeCollapse()` method that handles all
    // special cases is a lot of overhead; both, in terms of developing and
    // execution time.
    //
    // The main thing this method does is to check that the euler-
    // characteristic of the mesh doesn't change. As this is defined as
    // |V| - |E| + |F|, we have to make sure that an edge collapse would
    // remove as many edges (full edges!) as it removes vertices and faces
    // combined. This is true in the most general case.
    //
    // An edge collapse *always* removes one vertex, as well as one edge (those
    // two cancel out in Eulers formula). It also removes either one (edge is a
    // boundary edge) or two faces (general case). The special case of a lonely
    // edge is illegal for different reasons.
    //
    // The more difficult part is to figure out exactly how many edges would be
    // deleted by a collapse. An easy way to figure this out is the following:
    // consider the set `neighbors0` of all vertices that are adjacent to one
    // of the vertices connected to the collapsing edge. Consider the set
    // `neighbors1` for the other vertex as well. The subset `shared` of those
    // two sets contains all vertices which are directly connected to both
    // vertices of the collapsing edge.
    //
    // The number of vertices in this intersection set `shared` must equal the
    // number of adjacent faces. This is because for each shared vertex, we
    // fuse two edges, effectively removing one.
    //
    // Oh, and if you're wondering if it can actually happen that the number
    // of shared vertices is smaller than the number of faces: yes it can.
    // Imagine three vertices, three (full-)edges and two faces. The faces
    // point into opposite directions. Whether or not this mesh is broken is
    // already questionable; but it would be such a case.

    auto numFaces = numAdjacentFaces(handle);
    if (numFaces == 0)
    {
        // We don't allow collapsing lonely edges, as this can lead to
        // non-manifold vertices.
        return false;
    }

    // We also don't allow edge collapses that would lead to lonely edges. This
    // happens when a face adjacent to the collapsing edge has no adjacent
    // faces (other than the one which might be reached through the collapsing
    // edge itself, since that one is removed).
    for (auto faceH: getFacesOfEdge(handle))
    {
        if (faceH)
        {
            auto ns = getNeighboursOfFace(faceH.unwrap());

            // Subtract one face if the collapsing edge has two, because then
            // `ns` contains one of those two.
            auto numForeignNeighbours = ns.size() - (numFaces == 2 ? 1 : 0);

            if (numForeignNeighbours == 0)
            {
                return false;
            }
        }
    }

    // Obtain a list of neighbor vertices, as described above.
    auto vertices = getVerticesOfEdge(handle);
    auto neighbors0 = getNeighboursOfVertex(vertices[0]);
    auto neighbors1 = getNeighboursOfVertex(vertices[1]);

    // We don't explicitly store the `shared` set, but only count the number
    // of vertices in it. This is done with a stupid n² algorithm, but we
    // suspect the number of vertices to be very small.
    size_t sharedVerticesCount = std::count_if(neighbors0.begin(), neighbors0.end(), [&](auto v0)
    {
        return std::find(neighbors1.begin(), neighbors1.end(), v0) != neighbors1.end();
    });

    return sharedVerticesCount == numFaces;
}

template<typename BaseVecT>
bool BaseMesh<BaseVecT>::isFlippable(EdgeHandle handle) const
{
    if (numAdjacentFaces(handle) != 2)
    {
        return false;
    }

    // Make sure we have 4 different vertices around the faces of that edge.
    auto faces = getFacesOfEdge(handle);
    auto face0vertices = getVerticesOfFace(faces[0].unwrap());
    auto face1vertices = getVerticesOfFace(faces[1].unwrap());

    auto diffCount = std::count_if(face0vertices.begin(), face0vertices.end(), [&](auto f0v)
    {
        return std::find(face1vertices.begin(), face1vertices.end(), f0v) == face1vertices.end();
    });
    return diffCount == 1;
}

template<typename BaseVecT>
bool BaseMesh<BaseVecT>::isFaceInsertionValid(VertexHandle v1H, VertexHandle v2H, VertexHandle v3H) const
{
    // If there is already a face, we can't add another one
    if (getFaceBetween(v1H, v2H, v3H))
    {
        return false;
    }

    // Next we check that each vertex is a boundary one (it has at least one
    // boundary edge or no edges at all). We really need this property;
    // otherwise we would create non-manifold vertices and make the mesh non-
    // orientable.
    vector<EdgeHandle> edges;
    for (auto vH: {v1H, v2H, v3H})
    {
        edges.clear();
        getEdgesOfVertex(vH, edges);

        // No edges at all are fine too.
        if (edges.empty())
        {
            continue;
        }

        // But if there are some, we need at least one boundary one.
        auto boundaryEdgeIt = std::find_if(edges.begin(), edges.end(), [this](auto eH)
        {
            return this->numAdjacentFaces(eH) < 2;
        });
        if (boundaryEdgeIt == edges.end())
        {
            return false;
        }
    }

    return true;
}


template<typename BaseVecT>
OptionalFaceHandle BaseMesh<BaseVecT>::getFaceBetween(VertexHandle aH, VertexHandle bH, VertexHandle cH) const
{
    // Start with one edge
    const auto abEdgeH = getEdgeBetween(aH, bH);

    // If the edge already doesn't exist, there won't be a face either.
    if (!abEdgeH)
    {
        return OptionalFaceHandle();
    }

    // The two faces of the edge. One of these faces should contain the vertex
    // `c` or there is just no face between the given vertices.
    const auto faces = getFacesOfEdge(abEdgeH.unwrap());

    // Find the face which contains vertex `c`.
    auto faceIt = std::find_if(faces.begin(), faces.end(), [&, this](auto maybeFaceH)
    {
        if (!maybeFaceH)
        {
            return false;
        }
        const auto vertices = this->getVerticesOfFace(maybeFaceH.unwrap());
        return std::find(vertices.begin(), vertices.end(), cH) != vertices.end();
    });

    return faceIt != faces.end() ? faceIt->unwrap() : OptionalFaceHandle();
}


template<typename BaseVecT>
uint8_t BaseMesh<BaseVecT>::numAdjacentFaces(EdgeHandle handle) const
{
    auto faces = getFacesOfEdge(handle);
    return (faces[0] ? 1 : 0) + (faces[1] ? 1 : 0);
}

template<typename BaseVecT>
vector<FaceHandle> BaseMesh<BaseVecT>::getNeighboursOfFace(FaceHandle handle) const
{
    vector<FaceHandle> out;
    getNeighboursOfFace(handle, out);
    return out;
}

template<typename BaseVecT>
vector<FaceHandle> BaseMesh<BaseVecT>::getFacesOfVertex(VertexHandle handle) const
{
    vector<FaceHandle> out;
    getFacesOfVertex(handle, out);
    return out;
}

template<typename BaseVecT>
vector<EdgeHandle> BaseMesh<BaseVecT>::getEdgesOfVertex(VertexHandle handle) const
{
    vector<EdgeHandle> out;
    getEdgesOfVertex(handle, out);
    return out;
}

template<typename BaseVecT>
vector<VertexHandle> BaseMesh<BaseVecT>::getNeighboursOfVertex(VertexHandle handle) const
{
    vector<VertexHandle> out;
    getNeighboursOfVertex(handle, out);
    return out;
}

template<typename BaseVecT>
OptionalVertexHandle BaseMesh<BaseVecT>::getVertexBetween(EdgeHandle aH, EdgeHandle bH) const
{
    auto aEndpoints = getVerticesOfEdge(aH);
    auto bEndpoints = getVerticesOfEdge(bH);

    if (aEndpoints[0] == bEndpoints[0] || aEndpoints[0] == bEndpoints[1])
    {
        return aEndpoints[0];
    }
    if (aEndpoints[1] == bEndpoints[0] || aEndpoints[1] == bEndpoints[1])
    {
        return aEndpoints[1];
    }
    return OptionalVertexHandle();
}

template<typename BaseVecT>
OptionalEdgeHandle BaseMesh<BaseVecT>::getEdgeBetween(VertexHandle aH, VertexHandle bH) const
{
    // Go through all edges of vertex `a` until we find an edge that is also
    // connected to vertex `b`.
    for (auto eH: getEdgesOfVertex(aH))
    {
        auto endpoints = getVerticesOfEdge(eH);
        if (endpoints[0] == bH || endpoints[1] == bH)
        {
            return eH;
        }
    }
    return OptionalEdgeHandle();
}


} // namespace lvr2
