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

/**
 * PointsetSurface.tcc
 *
 *  @date 25.01.2012
 *  @author Thomas Wiemann
 */
namespace lvr2
{

template<typename BaseVecT>
PointsetSurface<BaseVecT>::PointsetSurface(PointBufferPtr<BaseVecT> pointBuffer)
    : m_pointBuffer(pointBuffer)
{
    // Calculate bounding box
    auto numPoints = m_pointBuffer->getNumPoints();
    for(size_t i = 0; i < numPoints; i++)
    {
        this->m_boundingBox.expand(m_pointBuffer->getPoint(i));
    }
}

template<typename BaseVecT>
Normal<BaseVecT> PointsetSurface<BaseVecT>::getInterpolatedNormal(Point<BaseVecT> position) const
{
    vector<size_t> indices;
    Vector<BaseVecT> result;
    m_searchTree->kSearch(position, m_ki, indices);
    for (int i = 0; i < m_ki; i++)
    {
        result += m_pointBuffer->getNormal(indices[i])->asVector();
    }
    result /= m_ki;
    return Normal<BaseVecT>(result);
}

template<typename BaseVecT>
const SearchTree<BaseVecT>& PointsetSurface<BaseVecT>::searchTree() const
{
    return *m_searchTree;
}

template<typename BaseVecT>
const BoundingBox<BaseVecT>& PointsetSurface<BaseVecT>::getBoundingBox() const
{
    return m_boundingBox;
}

template<typename BaseVecT>
PointBufferPtr<BaseVecT> PointsetSurface<BaseVecT>::pointBuffer() const
{
    return m_pointBuffer;
}

} // namespace lvr2