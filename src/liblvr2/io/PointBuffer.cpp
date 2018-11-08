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

#include <lvr2/io/PointBuffer.hpp>
#include <lvr2/io/Timestamp.hpp>

#include <iostream>

namespace lvr2
{

PointBuffer::PointBuffer()
{
    m_numPoints = 0;
}

PointBuffer::PointBuffer(floatArr points, size_t n)
{
    // Generate channel object pointer and add it
    // to channel map
    FloatChannelPtr point_data(new FloatChannel(n, 3, points));
    m_channels.addFloatChannel(point_data, "points");

    // Save pointers
    m_points = point_data;
    m_numPoints = n;
}

PointBuffer::PointBuffer(floatArr points, floatArr normals, size_t n) : PointBuffer(points, n)
{
    // Add normal data
    m_normals = FloatChannelPtr(new FloatChannel(n, 3, points));
    m_channels.addFloatChannel(m_normals, "normals");
}

void PointBuffer::setPointArray(floatArr points, size_t n)
{
    m_points = FloatChannelPtr(new FloatChannel(n, 3, points));
    m_numPoints = n;
    m_channels.addFloatChannel(m_points, "points");
}

void PointBuffer::setNormalArray(floatArr normals, size_t n)
{
    m_normals = FloatChannelPtr(new FloatChannel(n, 3, normals));
    m_channels.addFloatChannel(m_normals, "normals");
}
void PointBuffer::setColorArray(ucharArr colors, size_t n, unsigned width)
{
    m_colors = UCharChannelPtr(new UCharChannel(n, width, colors));
    m_channels.addUCharChannel(m_colors, "colors");
}

floatArr PointBuffer::getPointArray()
{
    if (m_points)
    {
        return m_points->dataPtr();
    }

    return floatArr();
}

floatArr PointBuffer::getNormalArray()
{
    if (m_normals)
    {
        return m_normals->dataPtr();
    }

    return floatArr();
}

ucharArr PointBuffer::getColorArray(unsigned& w)
{
    if (m_colors)
    {
        w = m_colors->width();
        return m_colors->dataPtr();
    }

    return ucharArr();
}


bool PointBuffer::hasColors() const
{
    if (m_colors)
    {
        return (m_colors->numAttributes() > 0);
    }

    return false;
}

bool PointBuffer::hasNormals() const
{
    if (m_normals)
    {
        return (m_normals->numAttributes() > 0);
    }

    return false;
}

size_t PointBuffer::numPoints() const
{
    return m_numPoints;
}



}


