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

#include <array>
#include <cstdio>
#include <memory>
#include <vector>

#include <boost/optional.hpp>

#include <lvr2/io/DataStruct.hpp>

#include <lvr2/geometry/Normal.hpp>
#include <lvr2/geometry/Vector.hpp>


using boost::optional;
using std::size_t;
using std::vector;
using std::array;


namespace lvr2
{

/**
 * \brief Stores points with various additional data channels.
 **/
template <typename BaseVecT>
class PointBuffer
{
public:
    PointBuffer() {}

    /**
     * \brief Get the number of points.
     **/
    size_t getNumPoints() const;

    //lvr::PointBuffer toOldBuffer() const;

    const Vector<BaseVecT>& getPoint(size_t idx) const;
    // Vector<BaseVecT>& getPoint(size_t idx);

    /**
     * @brief Returns true if the stored data contains normal information.
     */
    bool hasNormals() const;

    /**
     * @brief Adds (or overwrites) normal information for all points.
     *
     * All normals are initialized with the given value or with a dummy
     * value. Correct normals can later be set via `getNormal()`.
     */
    void addNormalChannel(Normal<BaseVecT> def = Normal<BaseVecT>(0, 0, 1));

    /**
     * @brief Copies normals from old buffer to this one.
     */
    void addNormalChannel(floatArr normals, size_t n);

    /**
     * @brief Returns the normal with the given index.
     */
    optional<const Normal<BaseVecT>&> getNormal(size_t idx) const;
    optional<Normal<BaseVecT>&> getNormal(size_t idx);

    /**
     * @brief Returns true if the stored data contains intensity information.
     */
    bool hasIntensities() const;

    /**
     * @brief Adds (or overwrites) intensity information for all points.
     *
     * All intensities are initialized with the given value or with 0. Correct
     * values can later be set via `getIntensity()`.
     */
    void addIntensityChannel(typename BaseVecT::CoordType def = 0);

    /**
     * @brief Returns the intensity with the given index.
     */
    optional<const typename BaseVecT::CoordType&> getIntensity(size_t idx) const;
    optional<typename BaseVecT::CoordType&> getIntensity(size_t idx);


    /**
     * @brief Returns true if the stored data contains confidence information.
     */
    bool hasConfidences() const;

    /**
     * @brief Adds (or overwrites) confidence information for all points.
     *
     * All confidences are initialized with the given value or with 0. Correct
     * values can later be set via `getConfidence()`.
     */
    void addConfidenceChannel(typename BaseVecT::CoordType def = 0);

    /**
     * @brief Returns the confidence with the given index.
     */
    optional<const typename BaseVecT::CoordType&> getConfidence(size_t idx) const;
    optional<typename BaseVecT::CoordType&> getConfidence(size_t idx);

    /**
     * @brief Returns true if the stored data contains RGB color information.
     */
    bool hasRgbColor() const;

    /**
     * @brief Adds (or overwrites) RGB color information for all points.
     *
     * All colors are initialized with the given value or with (0, 0, 0).
     * Correct values can later be set via `getRgbColor()`.
     */
    void addRgbColorChannel(array<uint8_t, 3> init = {0, 0, 0});

    /**
     * @brief Returns the RGB color with the given index.
     */
    optional<const array<uint8_t,3>&> getRgbColor(size_t idx) const;
    optional<array<uint8_t,3>&> getRgbColor(size_t idx);

    bool empty() const;

    void setPointArray(floatArr points, size_t);

    std::pair<floatArr, size_t> toFloatArr();

private:
    /// Point buffer.
    vector<Vector<BaseVecT>> m_points;

    /// Point normal buffer.
    optional<vector<Normal<BaseVecT>>> m_normals;

    /// Point intensity buffer.
    optional<vector<typename BaseVecT::CoordType>> m_intensities;

    /// Point confidence buffer.
    optional<vector<typename BaseVecT::CoordType>> m_confidences;

    /// Point RGB colors.
    optional<vector<array<uint8_t, 3>>> m_rgbColors;
};

template <typename BaseVecT>
using PointBufferPtr = std::shared_ptr<PointBuffer<BaseVecT>>;

} // namespace lvr2

#include <lvr2/io/PointBuffer.tcc>

#endif // LVR2_IO_POINTBUFFER_H_
