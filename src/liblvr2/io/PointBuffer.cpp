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


