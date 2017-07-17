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
    m_points.reserve(len);
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
        m_normals->reserve(normals_len);

        for (int i = 0; i < normals_len * 3; i += 3)
        {
            auto p = Normal<BaseVecT>(normal_buf[i], normal_buf[i + 1], normal_buf[i + 2]);
            m_normals->push_back(p);
        }
    }

    size_t intensities_len;
    auto intensities_buf = oldBuffer.getPointIntensityArray(intensities_len);
    if (intensities_len > 0)
    {
        m_intensities->reserve(intensities_len);
        std::copy(
            intensities_buf.get(),
            intensities_buf.get() + intensities_len,
            std::back_inserter(*m_intensities)
        );
    }

    size_t confidences_len;
    auto confidences_buf = oldBuffer.getPointConfidenceArray(confidences_len);
    if (confidences_len > 0)
    {
        m_confidences->reserve(confidences_len);
        std::copy(
            confidences_buf.get(),
            confidences_buf.get() + confidences_len,
            std::back_inserter(*m_confidences)
        );
    }

    size_t rgb_color_len;
    auto rgb_color_buf = oldBuffer.getPointColorArray(rgb_color_len);
    if (rgb_color_len > 0)
    {
        m_rgbColors = vector<array<uint8_t, 3>>();
        m_rgbColors->reserve(rgb_color_len);
        for (int i = 0; i < rgb_color_len * 3; i += 3)
        {
            m_rgbColors->push_back({
                rgb_color_buf[i],
                rgb_color_buf[i + 1],
                rgb_color_buf[i + 2]
            });
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

template <typename BaseVecT>
bool PointBuffer<BaseVecT>::hasIntensities() const {
    return static_cast<bool>(m_intensities);
}

template <typename BaseVecT>
void PointBuffer<BaseVecT>::addIntensityChannel(typename BaseVecT::CoordType def)
{
    m_intensities = vector<typename BaseVecT::CoordType>(getNumPoints(), def);
}

template <typename BaseVecT>
optional<const typename BaseVecT::CoordType&> PointBuffer<BaseVecT>::getIntensity(size_t idx) const
{
    if (!hasIntensities())
    {
        return boost::none;
    }
    return (*m_intensities)[idx];
}

template <typename BaseVecT>
optional<typename BaseVecT::CoordType&> PointBuffer<BaseVecT>::getIntensity(size_t idx)
{
    if (!hasIntensities())
    {
        return boost::none;
    }
    return (*m_intensities)[idx];
}

template <typename BaseVecT>
bool PointBuffer<BaseVecT>::hasConfidences() const {
    return static_cast<bool>(m_confidences);
}

template <typename BaseVecT>
void PointBuffer<BaseVecT>::addConfidenceChannel(typename BaseVecT::CoordType def)
{
    m_confidences = vector<typename BaseVecT::CoordType>(getNumPoints(), def);
}

template <typename BaseVecT>
optional<const typename BaseVecT::CoordType&> PointBuffer<BaseVecT>::getConfidence(size_t idx) const
{
    if (!hasConfidences())
    {
        return boost::none;
    }
    return (*m_confidences)[idx];
}

template <typename BaseVecT>
optional<typename BaseVecT::CoordType&> PointBuffer<BaseVecT>::getConfidence(size_t idx)
{
    if (!hasConfidences())
    {
        return boost::none;
    }
    return (*m_confidences)[idx];
}

template <typename BaseVecT>
bool PointBuffer<BaseVecT>::hasRgbColor() const
{
    return static_cast<bool>(m_rgbColors);
}

template <typename BaseVecT>
void PointBuffer<BaseVecT>::addRgbColorChannel(array<uint8_t, 3> init)
{
    m_rgbColors = vector<array<uint8_t, 3>>(getNumPoints(), init);
}

template <typename BaseVecT>
optional<const typename BaseVecT::CoordType&> PointBuffer<BaseVecT>::getRgbColor(size_t idx) const
{
    if (!hasRgbColor())
    {
        return boost::none;
    }
    return (*m_rgbColors)[idx];
}
template <typename BaseVecT>
optional<typename BaseVecT::CoordType&> PointBuffer<BaseVecT>::getRgbColor(size_t idx)
{
    if (!hasRgbColor())
    {
        return boost::none;
    }
    return (*m_rgbColors)[idx];
}

template <typename BaseVecT>
bool PointBuffer<BaseVecT>::empty() const
{
    return m_points.empty();
}

} // namespace lvr2
