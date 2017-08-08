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
    // The main thing this method does it to check that the euler-
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

} // namespace lvr2
