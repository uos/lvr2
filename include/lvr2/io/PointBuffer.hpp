/**
 * Copyright (c) 2019, University Osnabrück
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

#ifndef POINTBUFFER2_HPP
#define POINTBUFFER2_HPP

#include "lvr2/io/DataStruct.hpp"
#include "lvr2/types/BaseBuffer.hpp"

#include <map>
#include <string>

#include <boost/shared_array.hpp>
#include <iostream>

namespace lvr2
{

///
/// \brief A class to handle point information with an arbitrarily
///        large number of attribute channels. 
///        The added channels should always have the same length
///        as the point array to keep the mapping
///        between geometry (channel 'points') and the associated layers like RGB
///        colors or point normals consistent.
///
class PointBuffer : public BaseBuffer
{
    using base = BaseBuffer;
public:    
    PointBuffer();

    /***
     * @brief Constructs a point buffer with point the given number
     *        of point.
     *
     * @param points    An array containing point data (x,y,z).
     * @param n         Number of points
     */
    PointBuffer(floatArr points, size_t n);

    /***
     * @brief Constructs a point buffer with point and normal
     *        information. Both arrays are exspected to have the same
     *        length.
     *
     * @param points    An array containing point data (x,y,z).
     * @param normals   An array containing normal information (nx, ny, nz)
     * @param n         Number of points
     */
    PointBuffer(floatArr points, floatArr normals, size_t n);

    /***
     * @brief Adds points to the buffer. If the buffer already
     *        contains point cloud data, the interal buffer will
     *        be freed als well as all other attribute channels.
     */
    void setPointArray(floatArr points, size_t n);

    /***
     * @brief Adds an channel containing point normal data to the
     *        buffer.
     *
     * @param   normals A float array containing the normal data.
     *                  expected to be tuples (nx, ny, nz).
     */
    void setNormalArray(floatArr normals, size_t n);

    /***
     * @brief Generates and adds a channel for point color data
     *
     * @param   colors  Am array containing point cloud data
     * @param   n       Number of colors in the buffer
     * @param   width   Number of attributes per element. Normally
     *                  3 for RGB and 4 for RGBA buffers.
     */
    void setColorArray(ucharArr colors, size_t n, size_t width = 3);

    /// Returns the internal point array
    floatArr getPointArray();

    /// If the buffer stores normals, the
    /// call we return an empty array, i.e., the shared pointer
    /// contains a nullptr.
    floatArr getNormalArray();

    /// If the buffer stores color information, the
    /// call we return an empty array, i.e., the shared pointer
    /// contains a nullptr.
    ucharArr getColorArray(size_t& w);

    /// True, if buffer contains colors
    bool hasColors() const;

    /// True, if buffer has normals
    bool hasNormals() const;

    /// Returns the number of points in the buffer
    size_t numPoints() const;

    /// Makes a clone
    PointBuffer clone() const;

};

using PointBufferPtr = std::shared_ptr<PointBuffer>;

} // namespace lvr2

#endif // POINTBUFFER2_HPP
