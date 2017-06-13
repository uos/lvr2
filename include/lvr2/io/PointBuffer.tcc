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
 * @file      PointBuffer.tcc
 *
 **/


#include <cstring>

namespace lvr2
{

PointBuffer::PointBuffer(lvr::PointBuffer& oldBuffer)
{
    // This method is temporary only, until the old `PointBuffer` can be
    // discarded.
    size_t len;
    auto buf = oldBuffer.getPointArray(len);
    m_points.resize(len * 4);
    memcpy(m_points.data(), buf.get(), len * 4);
}


template <typename BaseVecT>
size_t PointBuffer::getNumPoints() const
{
    return m_points.size() / (sizeof(typename BaseVecT::CoordType) * 3);
}

template <typename BaseVecT>
Point<BaseVecT> PointBuffer::getPoint(size_t idx) const
{
    return Point<BaseVecT>(getBaseVec<BaseVecT>(idx));
}

template <typename BaseVecT>
optional<Normal<BaseVecT>> PointBuffer::getNormal(size_t idx) const
{
    if (!hasNormals())
    {
        return boost::none;
    }
    return Normal<BaseVecT>(getBaseVec<BaseVecT>(idx));
}

template <typename BaseVecT>
BaseVecT PointBuffer::getBaseVec(size_t idx) const
{
    using CoordT = typename BaseVecT::CoordType;

    // This class currently only supports coordinate types that don't require
    // an alignment bigger than 8.
    static_assert(alignof(CoordT) <= 8, "unsupported vector type");

    auto step = sizeof(CoordT);
    auto offset = idx * 3;
    auto x = *reinterpret_cast<const CoordT*>(&m_points[offset + 0 * step]);
    auto y = *reinterpret_cast<const CoordT*>(&m_points[offset + 1 * step]);
    auto z = *reinterpret_cast<const CoordT*>(&m_points[offset + 2 * step]);

    return BaseVecT(x, y, z);
}

bool PointBuffer::empty() const {
    return m_points.empty();
}

bool PointBuffer::hasNormals() const {
    return !m_normals.empty();
}

} // namespace lvr2
