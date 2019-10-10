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
 * Util.hpp
 *
 *  @date 30.10.2018
 *  @author Alexander Loehr (aloehr@uos.de)
 */

#ifndef LVR2_UTIL_HPP
#define LVR2_UTIL_HPP

#include <vector>
#include <boost/shared_array.hpp>

#include "lvr2/types/MatrixTypes.hpp"
#include "lvr2/io/PointBuffer.hpp"
#include "lvr2/geometry/BaseVector.hpp"
#include "lvr2/geometry/Matrix4.hpp"


namespace lvr2
{

using VecUChar = BaseVector<unsigned char>;

/**
 * @brief A class that contains utility functions/types
 *
 */
class Util
{
public:

    /**
     * @brief Returns the spectral channel index for a given wavelength.
     *
     * @param   wavelength    The wavenlength
     * @param   p             A pointcloud pointer
     * @param   fallback      A fallback that will be returned if something went
     *                        wrong (default -1)
     *
     * @return Returns for a wavelength the corresponding spectral channel index for the
     *         pointcloud or fallback if the pointcloud has no spectral
     *         channel data or the wavelength is not in the given range
     *         of the spectral channel data
     *
     */
    static int getSpectralChannel(int wavelength, PointBufferPtr p, int fallback = -1);

    /**
     * @brief For a given spectral channel it return the corresponding wavelength.
     *
     * @param   channel    The spectral channel index
     * @param   p          A pointcloud pointer
     * @param   fallback   A fallback that will be returned if something went wrong (default -1)
     *
     * @return Returns for a spectral channel index  the corresponding wavelength for
     *         the given pointcloud or fallback if the pointcloud has no spectral channel
     *         data or the spectral channel index is out of range.
     *
     */
    static int getSpectralWavelength(int channel, PointBufferPtr p, int fallback = -1);

    /**
     * @brief Calculates the wavelength distance between two spectral channels.
     *
     * @param    p    The pointcloud pointer
     *
     * @return Returns the wavelength distance between two spectral channels
     *         and if the pointcloud has no spectral channel data it return -1
     */
    static float wavelengthPerChannel(PointBufferPtr p);

    /**
     * @brief Creates a shared array with the data from the given vector.
     *
     * @param   source    The vector from where the data will be copied
     *
     * @return returns A shared array with a copy of the data from the given vector
     */
    template<typename T>
    static boost::shared_array<T> convert_vector_to_shared_array(std::vector<T> source)
    {
        boost::shared_array<T> ret = boost::shared_array<T>( new T[source.size()] );
        std::copy(source.begin(), source.end(), ret.get());

        return ret;
    }

    /**
     * @brief Converts a transformation matrix that is used in riegl coordinate system into
     *        a transformation matrix that is used in slam6d coordinate system.
     *
     * @param    in    The transformation matrix in riegl coordinate system
     *
     * @return The transformation matrix in slam6d coordinate system
     */
    template <typename BaseVecT>
    static Matrix4<BaseVecT> riegl_to_slam6d_transform(const Matrix4<BaseVecT> &in)
    {
        Matrix4<BaseVecT> ret;

        ret[0] = in[5];
        ret[1] = -in[9];
        ret[2] = -in[1];
        ret[3] = -in[13];
        ret[4] = -in[6];
        ret[5] = in[10];
        ret[6] = in[2];
        ret[7] = in[14];
        ret[8] = -in[4];
        ret[9] = in[8];
        ret[10] = in[0];
        ret[11] = in[12];
        ret[12] = -100*in[7];
        ret[13] = 100*in[11];
        ret[14] = 100*in[3];
        ret[15] = in[15];

        return ret;
    }

    /**
     * @brief Converts a transformation matrix that is used in slam6d coordinate system into
     *        a transformation matrix that is used in riegl coordinate system.
     *
     * @param    in    The transformation matrix in slam6d coordinate system
     *
     * @return The transformation matrix in riegl coordinate system
     */
    template <typename T>
    Transform<T> slam6d_to_riegl_transform(const Transform<T> &mat)
    {
        T* in = mat.data();
        T* ret[16];
        ret[0] = in[10];
        ret[1] = -in[2];
        ret[2] = in[6];
        ret[3] = in[14]/100.0;
        ret[4] = -in[8];
        ret[5] = in[0];
        ret[6] = -in[4];
        ret[7] = -in[12]/100.0;
        ret[8] = in[9];
        ret[9] = -in[1];
        ret[10] = in[5];
        ret[11] = in[13]/100.0;
        ret[12] = in[11];
        ret[13] = -in[3];
        ret[14] = in[7];
        ret[15] = in[15];

        return Eigen::Map<Eigen::Matrix<T, 4, 4, Eigen::RowMajor> >(ret);
    }

    template <typename ValueType>
    static BaseVector<ValueType> slam6d_to_riegl_point(const BaseVector<ValueType> &in)
    {
        return {
            in.z   / (ValueType) 100.0,
            - in.x / (ValueType) 100.0,
            in.y   / (ValueType) 100.0
        };
    }



    /**
     * @brief Converts an angle from degree to radian.
     *
     * @param    deg    Angle in degree
     *
     * @return Angle in radians
     */
    template <typename ValueType>
    static ValueType deg_to_rad(ValueType deg)
    {
        return M_PI / 180.0 * deg;
    }

    /**
     * @brief Converts an angle from radian to degree.
     *
     * @param    rad    Angle in radians
     *
     * @return Angle in degree
     */
    template <typename ValueType>
    static ValueType rad_to_deg(ValueType rad)
    {
        return rad * 180 / M_PI;
    }

    /**
     * @brief A comparison object for Vector<VecUChar>
     */
    struct ColorVecCompare
    {

        /**
         * @brief Comparison operator
         *
         * @param    lhs    The first object for comparison
         * @param    rhs    The second object for comparison
         *
         * @return Returns true if lhs is smaller and elsewise false
         */
        bool operator() (const VecUChar& lhs, const VecUChar& rhs) const
        {
            return (lhs.x < rhs.x) ||
                   (lhs.x == rhs.x && lhs.y < rhs.y) ||
                   (lhs.x == rhs.x && lhs.y == rhs.y && lhs.z < rhs.z);
        }
    };
};

} // namespace lvr2

#endif
