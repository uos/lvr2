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

#ifndef POLYGONBUFFER
#define POLYGONBUFFER

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
class PolygonBuffer : public BaseBuffer
{
    using base = BaseBuffer;
public:
    PolygonBuffer();

    /***
     * @brief Constructs a point Polygon with point the given number
     *        of point.
     *
     * @param points    An array containing point data (x,y,z).
     * @param n         Number of points
     */
    PolygonBuffer(floatArr points, size_t n);

    PolygonBuffer(std::vector<std::vector<float> >& points);
    /***
     * @brief Adds points to the buffer. If the buffer already
     *        contains point cloud data, the interal buffer will
     *        be freed als well as all other attribute channels.
     */
    void setPointArray(floatArr points, size_t n);


    /// Todo: remove once normal HDF IO implemented
    void load(std::string file);

    /// Returns the internal point array
    floatArr getPointArray();

    /// Returns the number of points in the buffer
    size_t numPoints() const;

    /// Makes a clone
    PolygonBuffer clone() const;

};

using PolygonBufferPtr = std::shared_ptr<PolygonBuffer>;

} // namespace lvr2

#endif // POLYGONBUFFER
