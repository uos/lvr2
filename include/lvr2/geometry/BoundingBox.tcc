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
 * BoundingBox.tcc
 *
 *  @date 22.10.2008
 *  @author Thomas Wiemann (twiemann@uos.de)
 */

#include <algorithm>
#include <limits>

using std::numeric_limits;

#include "lvr2/io/LineReader.hpp"

namespace lvr2
{

template<typename BaseVecT>
BoundingBox<BaseVecT>::BoundingBox()
{
    auto max_val = numeric_limits<typename BaseVecT::CoordType>::max();
    auto min_val = numeric_limits<typename BaseVecT::CoordType>::lowest();

    m_min = BaseVecT(max_val, max_val, max_val);
    m_max = BaseVecT(min_val, min_val, min_val);
}

template<typename BaseVecT>
template<typename T>
BoundingBox<BaseVecT>::BoundingBox(T v1, T v2)
{
    m_min = v1;
    m_max = v2;
    
    m_centroid = BaseVecT(
        m_min.x + 0.5f * getXSize(),
        m_min.y + 0.5f * getYSize(),
        m_min.z + 0.5f * getZSize()
    );

}

template<typename BaseVecT>
BoundingBox<BaseVecT>::BoundingBox(std::string plyPath)
{
    auto max_val = numeric_limits<typename BaseVecT::CoordType>::max();
    auto min_val = numeric_limits<typename BaseVecT::CoordType>::lowest();

    m_min = BaseVecT(max_val, max_val, max_val);
    m_max = BaseVecT(min_val, min_val, min_val);

    size_t rsize = 0;
    LineReader lineReader(plyPath);
    size_t lasti = 0;
    while (lineReader.ok())
    {
        if (lineReader.getFileType() == XYZNRGB)
        {
            boost::shared_ptr<xyznc> a = boost::static_pointer_cast<xyznc>(
                    lineReader.getNextPoints(rsize, 1024));
            if (rsize <= 0 && !lineReader.ok())
            {
                break;
            }
            for (int i = 0; i < rsize; i++)
            {
                //original: multiplied by scale
                expand(BaseVecT(a.get()[i].point.x * 1,
                        a.get()[i].point.y * 1,
                        a.get()[i].point.z * 1));

            }
        }
        else if (lineReader.getFileType() == XYZN)
        {
            boost::shared_ptr<xyzn> a = boost::static_pointer_cast<xyzn>(
                    lineReader.getNextPoints(rsize, 1024));
            if (rsize <= 0 && !lineReader.ok())
            {
                break;
            }
            for (int i = 0; i < rsize; i++)
            {
                expand(BaseVecT(a.get()[i].point.x *1,
                        a.get()[i].point.y * 1,
                        a.get()[i].point.z * 1));

            }
        }
        else if (lineReader.getFileType() == XYZ)
        {
            boost::shared_ptr<xyz> a =
                    boost::static_pointer_cast<xyz>(lineReader.getNextPoints(rsize, 1024));
            if (rsize <= 0 && !lineReader.ok())
            {
                break;
            }
            for (size_t i = 0; i < rsize; i++)
            {
                expand(BaseVecT(a.get()[i].point.x * 1,
                        a.get()[i].point.y * 1,
                        a.get()[i].point.z * 1));

                lasti = i;
            }
        }
        else if (lineReader.getFileType() == XYZRGB)
        {
            boost::shared_ptr<xyzc> a = boost::static_pointer_cast<xyzc>(
                    lineReader.getNextPoints(rsize, 1024));
            if (rsize <= 0 && !lineReader.ok())
            {
                break;
            }
            for (size_t i = 0; i < rsize; i++)
            {
                expand(BaseVecT(a.get()[i].point.x * 1,
                        a.get()[i].point.y * 1,
                        a.get()[i].point.z * 1));
                lasti = i;
            }
        }
        else
        {
            exit(-1);
        }
    }
}

template<typename BaseVecT>
bool BoundingBox<BaseVecT>::isValid() const
{
    return m_min.x < m_max.x
        && m_min.y < m_max.y
        && m_min.z < m_max.z;
}

template<typename BaseVecT>
bool BoundingBox<BaseVecT>::overlap(const lvr2::BoundingBox<BaseVecT> &bb)
{
    return (m_min.x <= bb.m_max.x && m_max.x >= bb.m_min.x)
        && (m_min.y <= bb.m_max.y && m_max.y >= bb.m_min.y)
        && (m_min.z <= bb.m_max.z && m_max.z >= bb.m_min.z);
}

template<typename BaseVecT>
BaseVecT BoundingBox<BaseVecT>::getCentroid() const
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
template<typename T>
inline void BoundingBox<BaseVecT>::expand(T v)
{
    m_min.x = std::min(v.x, m_min.x);
    m_min.y = std::min(v.y, m_min.y);
    m_min.z = std::min(v.z, m_min.z);

    m_max.x = std::max(v.x, m_max.x);
    m_max.y = std::max(v.y, m_max.y);
    m_max.z = std::max(v.z, m_max.z);

    m_centroid = BaseVecT(
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
typename BaseVecT::CoordType BoundingBox<BaseVecT>::getVolume() const
{
    return getXSize() * getYSize() * getZSize();
}

template<typename BaseVecT>
typename BaseVecT::CoordType BoundingBox<BaseVecT>::getZSize() const
{
    return m_max.z - m_min.z;
}

template<typename BaseVecT>
BaseVecT BoundingBox<BaseVecT>::getMin() const
{
    return m_min;
}


template<typename BaseVecT>
BaseVecT BoundingBox<BaseVecT>::getMax() const
{
    return m_max;
}


} // namespace lvr2
