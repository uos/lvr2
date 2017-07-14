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

template <typename BaseVecT>
PointBuffer<BaseVecT>::PointBuffer(lvr::PointBuffer& oldBuffer)
{
    // This method is temporary only, until the old `PointBuffer` can be
    // discarded.
    size_t len;
    auto buf = oldBuffer.getPointArray(len);
    for (int i = 0; i < len * 3; i += 3)
    {
        auto p = Point<BaseVecT>(buf[i], buf[i + 1], buf[i + 2]);
        m_points.push_back(p);
    }

    if (oldBuffer.hasPointNormals())
    {
        m_normals = vector<Normal<BaseVecT>>();
        size_t normals_len;
        auto normal_buf = oldBuffer.getPointNormalArray(normals_len);
        for (int i = 0; i < normals_len * 3; i += 3)
        {
            auto p = Normal<BaseVecT>(normal_buf[i], normal_buf[i + 1], normal_buf[i + 2]);
            m_normals->push_back(p);
        }
    }
}


template <typename BaseVecT>
size_t PointBuffer<BaseVecT>::getNumPoints() const
{
    return m_points.size();
}

template <typename BaseVecT>
lvr::PointBuffer PointBuffer<BaseVecT>::toOldBuffer() const
{
    auto out = lvr::PointBuffer();

    auto pointData = boost::shared_array<float>(new float[m_points.size() * 3]);
    for (size_t i = 0; i < m_points.size(); i++) {
        auto p = m_points[i];
        pointData[i + 0] = p.x;
        pointData[i + 1] = p.y;
        pointData[i + 2] = p.z;
    }
    out.setPointArray(pointData, m_points.size());

    if (m_normals)
    {
        auto normalData = boost::shared_array<float>(new float[m_normals->size() * 3]);
        for (size_t i = 0; i < m_normals->size(); i++) {
            auto p = (*m_normals)[i];
            normalData[i + 0] = p.getX();
            normalData[i + 1] = p.getY();
            normalData[i + 2] = p.getZ();
        }
        out.setPointNormalArray(normalData, m_normals->size());
    }

    // TODO the remaining stuff

    return out;
}


template <typename BaseVecT>
const Point<BaseVecT>& PointBuffer<BaseVecT>::getPoint(size_t idx) const
{
    return m_points[idx];
}

// template <typename BaseVecT>
// Point<BaseVecT>& PointBuffer<BaseVecT>::getPoint(size_t idx)
// {
//     return m_points[idx];
// }


template <typename BaseVecT>
bool PointBuffer<BaseVecT>::hasNormals() const {
    return static_cast<bool>(m_normals);
}

template <typename BaseVecT>
void PointBuffer<BaseVecT>::addNormalChannel(Normal<BaseVecT> def)
{
    m_normals = vector<Normal<BaseVecT>>(getNumPoints(), def);
}

template <typename BaseVecT>
optional<const Normal<BaseVecT>&> PointBuffer<BaseVecT>::getNormal(size_t idx) const
{
    if (!hasNormals())
    {
        return boost::none;
    }
    return (*m_normals)[idx];
}

template <typename BaseVecT>
optional<Normal<BaseVecT>&> PointBuffer<BaseVecT>::getNormal(size_t idx)
{
    if (!hasNormals())
    {
        return boost::none;
    }
    return (*m_normals)[idx];
}

// template <typename BaseVecT>
// BaseVecT PointBuffer<BaseVecT>::getBaseVec(size_t idx) const
// {
//     using CoordT = typename BaseVecT::CoordType;

//     // This class currently only supports coordinate types that don't require
//     // an alignment bigger than 8.
//     static_assert(alignof(CoordT) <= 8, "unsupported vector type");

//     auto step = sizeof(CoordT);
//     auto offset = idx * 3;
//     auto x = *reinterpret_cast<const CoordT*>(&m_points[offset + 0 * step]);
//     auto y = *reinterpret_cast<const CoordT*>(&m_points[offset + 1 * step]);
//     auto z = *reinterpret_cast<const CoordT*>(&m_points[offset + 2 * step]);

//     return BaseVecT(x, y, z);
// }

template <typename BaseVecT>
bool PointBuffer<BaseVecT>::empty() const {
    return m_points.empty();
}

} // namespace lvr2
