#include <lvr2/io/PointBuffer2.hpp>
#include <lvr2/io/Timestamp.hpp>

#include <iostream>

namespace lvr2
{

PointBuffer2::PointBuffer2()
{
    m_numPoints = 0;
}

PointBuffer2::PointBuffer2(floatArr points, size_t n)
{
    // Generate channel object pointer and add it
    // to channel map
    FloatChannelPtr point_data(new FloatChannel(n, 3, points));
    m_channels.addFloatChannel(point_data, "points");

    // Save pointers
    m_points = point_data;
    m_numPoints = n;
}

PointBuffer2::PointBuffer2(floatArr points, floatArr normals, size_t n) : PointBuffer2(points, n)
{
    // Add normal data
    FloatChannelPtr normal_data(new FloatChannel(n, 3, points));
    m_channels.addFloatChannel(normal_data, "normals");
}

void PointBuffer2::setPointArray(floatArr points, size_t n)
{
    m_points = FloatChannelPtr(new FloatChannel(n, 3, points));
    m_numPoints = n;
    m_channels.addFloatChannel(m_points, "points");
}

void PointBuffer2::setNormalArray(floatArr normals, size_t n)
{
    m_normals = FloatChannelPtr(new FloatChannel(n, 3, normals));
    m_channels.addFloatChannel(m_normals, "normals");
}
void PointBuffer2::setColorArray(ucharArr colors, size_t n, unsigned width)
{
    m_colors = UCharChannelPtr(new UCharChannel(n, width, colors));
    m_channels.addUCharChannel(m_colors, "colors");
}

floatArr PointBuffer2::getPointArray()
{
    return m_points->dataPtr();
}

floatArr PointBuffer2::getNormalArray()
{
    return m_normals->dataPtr();
}

ucharArr PointBuffer2::getColorArray(unsigned& w)
{
    w = m_colors->width();
    return m_colors->dataPtr();
}


bool PointBuffer2::hasColors() const
{
    if (m_colors)
    {
        return (m_colors->numAttributes() > 0);
    }

    return false;
}

bool PointBuffer2::hasNormals() const
{
    if (m_normals)
    {
        return (m_normals->numAttributes() > 0);
    }

    return false;
}

size_t PointBuffer2::numPoints() const
{
    return m_numPoints;
}



}


