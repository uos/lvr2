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

#include "lvr2/io/PointBuffer.hpp"
#include "lvr2/io/Timestamp.hpp"

#include <iostream>

namespace lvr2
{

PointBuffer::PointBuffer()
:base()
{
   
}

PointBuffer::PointBuffer(floatArr points, size_t n)
{
    // Generate channel object pointer and add it
    // to channel map
    FloatChannelPtr point_data(new FloatChannel(n, 3, points));
    this->addFloatChannel(point_data, "points");

}

PointBuffer::PointBuffer(floatArr points, floatArr normals, size_t n) : PointBuffer(points, n)
{
    // Add normal data
    FloatChannelPtr normal_data(new FloatChannel(n, 3, points));
    this->addFloatChannel(normal_data, "normals");
}

void PointBuffer::setPointArray(floatArr points, size_t n)
{
    FloatChannelPtr pts(new FloatChannel(n, 3, points));
    this->addFloatChannel(pts, "points");
}

void PointBuffer::setNormalArray(floatArr normals, size_t n)
{
    FloatChannelPtr nmls(new FloatChannel(n, 3, normals));
    this->addFloatChannel(nmls, "normals");
}
void PointBuffer::setColorArray(ucharArr colors, size_t n, size_t width)
{
    UCharChannelPtr cls(new UCharChannel(n, width, colors));
    this->addUCharChannel(cls, "colors");
}

floatArr PointBuffer::getPointArray()
{
    typename Channel<float>::Optional opt = getChannel<float>("points");
    if(opt)
    {
        return opt->dataPtr();
    }

    return floatArr();
}

floatArr PointBuffer::getNormalArray()
{
    typename Channel<float>::Optional opt = getChannel<float>("normals");
    if(opt)
    {
        return opt->dataPtr();
    }

    return floatArr();
}

ucharArr PointBuffer::getColorArray(size_t& w)
{
    w = 0;
    typename Channel<unsigned char>::Optional opt = getChannel<unsigned char>("colors");
    if(opt)
    {
        w = opt->width();
        return opt->dataPtr();
    }

    return ucharArr();
}


bool PointBuffer::hasColors() const
{
   return hasChannel<unsigned char>("colors");
}

bool PointBuffer::hasNormals() const
{
   return hasChannel<float>("normals");
}

size_t PointBuffer::numPoints() const
{
    const typename Channel<float>::Optional opt = getChannel<float>("points");
    if(opt)
    {
        return opt->numElements();
    }
    else
    {
        return 0;
    }
    
}


PointBuffer PointBuffer::clone() const
{
    PointBuffer pb;

    for(const auto& elem : *this)
    {
        pb.insert({elem.first, elem.second.clone()});
    }

    return pb;

}



}


