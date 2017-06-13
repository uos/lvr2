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
 *
 * @file      PointLoader.hpp
 * @brief     Interface for all point loading classes.
 * @details   The PointLoader class specifies storage and access to all
 *            available point data by implementing the appropriate  get and set
 *            methods for these data.
 *
 * @author    Lars Kiesow (lkiesow), lkiesow@uos.de, Universität Osnabrück
 * @author    Thomas Wiemann, twiemann@uos.de, Universität Osnabrück
 * @version   111001
 * @date      Recreated:     2011-09-22 23:23:57
 * @date      Last modified: 2011-10-01 15:22:49
 *
 **/

#ifndef LVR2_IO_POINTBUFFER_H_
#define LVR2_IO_POINTBUFFER_H_

// #include <stdint.h>
// #include <cstddef>
// #include <cstdlib>

// #include <gsl/gsl>

#include <vector>
#include <cstdio>

#include <boost/optional.hpp>

#include <lvr/io/PointBuffer.hpp>

#include <lvr2/geometry/Normal.hpp>
#include <lvr2/geometry/Point.hpp>


using boost::optional;
using std::size_t;


namespace lvr2
{

/**
 * \brief Stores points with various additional data channels.
 **/
class PointBuffer
{
public:
    PointBuffer(lvr::PointBuffer& oldBuffer);

    /**
     * \brief Get the number of points.
     **/
    template <typename BaseVecT>
    size_t getNumPoints() const;

    template <typename BaseVecT>
    Point<BaseVecT> getPoint(size_t idx) const;

    template <typename BaseVecT>
    optional<Normal<BaseVecT>> getNormal(size_t idx) const;

    /**
     * @brief   Returns true if the stored data contains point normal
     *          information
     */
    bool hasNormals() const;


private:
    struct alignas(8) AlignedByte
    {
        uint8_t data;
    };

    /// Point buffer.
    vector<AlignedByte> m_points;

    /// Point normal buffer.
    vector<AlignedByte> m_normals;

    /// Point intensity buffer.
    vector<AlignedByte> m_intensities;

    /// Point confidence buffer.
    vector<AlignedByte> m_confidences;
};

typedef boost::shared_ptr<PointBuffer> PointBufferPtr;

} // namespace lvr2

#include <lvr2/io/PointBuffer.tcc>

#endif // LVR2_IO_POINTBUFFER_H_
