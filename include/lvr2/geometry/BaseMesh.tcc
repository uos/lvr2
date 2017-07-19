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
Point<BaseVecT> BaseMesh<BaseVecT>::calcFaceCentroid(FaceHandle handle) const
{
    auto points = this->getVertexPositionsOfFace(handle);

    Vector<BaseVecT> centroid;
    for (auto point: points)
    {
        centroid += point.asVector();
    }
    centroid /= points.size();

    // Convert to point
    return Point<BaseVecT>(centroid);
}

} // namespace lvr2
