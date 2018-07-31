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
void PointBuffer2::setColorArray(ucharArr colors, size_t n)
{
    m_colors = UCharChannelPtr(new UCharChannel(n, 3, colors));
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

floatArr PointBuffer2::getFloatArray(const std::string& name, unsigned& w)
{
    size_t n;
    floatArr arr = m_channels.getFloatArray(n, w, name);

    if(n != m_numPoints)
    {
        std::cout << timestamp << "PointBuffer::getFloatArray(): Size mismatch for attribute '"
                  << name <<"': " << m_numPoints << " / " << n << std::endl;
    }
    return arr;
}

ucharArr PointBuffer2::getColorArray()
{
    return m_colors->dataPtr();
}

ucharArr PointBuffer2::getUcharArray(const std::string& name, unsigned& w)
{
    size_t n;
    ucharArr arr = m_channels.getUCharArray(n, w, name);

    if(n != m_numPoints)
    {
        std::cout << timestamp << "PointBuffer::getUCharArray(): Size mismatch for attribute '"
                  << name <<"': " << m_numPoints << " / " << n << std::endl;
    }
    return arr;
}

FloatChannel PointBuffer2::getFloatChannel(const std::string& name)
{
    return m_channels.getFloatChannel(name);
}

UCharChannel PointBuffer2::getUcharChannel(const std::string& name)
{
    return m_channels.getUCharChannel(name);
}

bool PointBuffer2::hasColors() const
{
    return (m_colors->numAttributes() > 0);
}

bool PointBuffer2::hasNormals() const
{
    return (m_normals->numAttributes() > 0);
}

size_t PointBuffer2::numPoints() const
{
    return m_numPoints;
}

void PointBuffer2::addFloatChannel(floatArr data, std::string name, size_t n, unsigned w)
{
    m_channels.addFloatChannel(data, name, n, w);
}


void PointBuffer2::addUCharChannel(ucharArr data, std::string name, size_t n, unsigned w)
{
    m_channels.addUCharChannel(data, name, n, w);
}


}


