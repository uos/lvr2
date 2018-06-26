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
 * BoundingBox.tcc
 *
 *  @date 22.10.2008
 *  @author Thomas Wiemann (twiemann@uos.de)
 */

#include <algorithm>
#include <limits>

using std::numeric_limits;

namespace lvr2
{

template<typename BaseVecT>
BoundingBox<BaseVecT>::BoundingBox()
{
    auto max_val = numeric_limits<typename BaseVecT::CoordType>::max();
    auto min_val = numeric_limits<typename BaseVecT::CoordType>::min();

    m_min = Point<BaseVecT>(max_val, max_val, max_val);
    m_max = Point<BaseVecT>(min_val, min_val, min_val);
}

template<typename BaseVecT>
BoundingBox<BaseVecT>::BoundingBox(Point<BaseVecT> v1, Point<BaseVecT> v2)
{
    m_min = v1;
    m_max = v2;
    
    m_centroid = Point<BaseVecT>(
        m_min.x + 0.5f * getXSize(),
        m_min.y + 0.5f * getYSize(),
        m_min.z + 0.5f * getZSize()
    );

}

template<typename BaseVecT>
bool BoundingBox<BaseVecT>::isValid() const
{
    return m_min.x < m_max.x
        && m_min.y < m_max.y
        && m_min.z < m_max.z;
}

template<typename BaseVecT>
Point<BaseVecT> BoundingBox<BaseVecT>::getCentroid() const
{
    return m_centroid;
}

template<typename BaseVecT>
typename BaseVecT::CoordType BoundingBox<BaseVecT>::getRadius() const
{
    // Shift bounding box to (0,0,0)
    auto m_min0 = m_min - m_centroid;
    auto m_max0 = m_max - m_centroid;

    return std::max(m_min0.length(), m_max0.length());
}

template<typename BaseVecT>
inline void BoundingBox<BaseVecT>::expand(Point<BaseVecT> v)
{
    m_min.x = std::min(v.x, m_min.x);
    m_min.y = std::min(v.y, m_min.y);
    m_min.z = std::min(v.z, m_min.z);

    m_max.x = std::max(v.x, m_max.x);
    m_max.y = std::max(v.y, m_max.y);
    m_max.z = std::max(v.z, m_max.z);

    m_centroid = Point<BaseVecT>(
        m_min.x + 0.5f * getXSize(),
        m_min.y + 0.5f * getYSize(),
        m_min.z + 0.5f * getZSize()
    );
}

template<typename BaseVecT>
inline void BoundingBox<BaseVecT>::expand(const BoundingBox<BaseVecT>& bb)
{
    expand(bb.m_min);
    expand(bb.m_max);
}

template<typename BaseVecT>
typename BaseVecT::CoordType BoundingBox<BaseVecT>::getLongestSide() const
{
    return std::max({ getXSize(), getYSize(), getZSize() });
}

template<typename BaseVecT>
typename BaseVecT::CoordType BoundingBox<BaseVecT>::getXSize() const
{
    return m_max.x - m_min.x;
}

template<typename BaseVecT>
typename BaseVecT::CoordType BoundingBox<BaseVecT>::getYSize() const
{
    return m_max.y - m_min.y;
}


template<typename BaseVecT>
typename BaseVecT::CoordType BoundingBox<BaseVecT>::getZSize() const
{
    return m_max.z - m_min.z;
}

template<typename BaseVecT>
Point<BaseVecT> BoundingBox<BaseVecT>::getMin() const
{
    return m_min;
}


template<typename BaseVecT>
Point<BaseVecT> BoundingBox<BaseVecT>::getMax() const
{
    return m_max;
}


} // namespace lvr2
