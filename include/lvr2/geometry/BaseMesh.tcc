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
    // The answer at [1] describes in more detail, what we need to check. Note
    // however, that we don't need to check whether or not the normal will
    // flip. This does not have negative consequences for 3D meshes
    //
    // [1]: https://stackoverflow.com/a/27049418/2408867
    auto vertices = getVerticesOfEdge(handle);
    auto neighbors0 = getNeighboursOfVertex(vertices[0]);
    auto neighbors1 = getNeighboursOfVertex(vertices[1]);

    size_t sharedVerticesCount = 0;
    for (auto v0: neighbors0)
    {
        if (std::find(neighbors1.begin(), neighbors1.end(), v0) != neighbors1.end())
        {
            sharedVerticesCount += 1;
            if (sharedVerticesCount > 2)
            {
                return false;
            }
        }
    }

    return true;
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
