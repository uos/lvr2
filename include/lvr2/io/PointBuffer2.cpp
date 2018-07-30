#include <lvr2/io/PointBuffer2.hpp>
#include <lvr2/io/Timestamp.hpp>

#include <iostream>

namespace lvr2
{

PointBuffer2::PointBuffer2(floatArr points, size_t n)
{
    // Generate channel object pointer and add it
    // to channel map
    FloatChannelPtr point_data(new FloatChannel(n, 3, points));
    addFloatChannel(point_data, "points");

    // Save pointers
    m_points = point_data;
    m_numPoints = n;
}

PointBuffer2::PointBuffer2(floatArr points, floatArr normals, size_t n) : PointBuffer2(points, n)
{
    // Add normal data
    FloatChannelPtr normal_data(new FloatChannel(n, 3, points));
    addFloatChannel(normal_data, "normals");
}

void PointBuffer2::createFloatChannel(FloatChannelPtr data, std::string name, size_t n, unsigned width)
{
    floatArr array(new float(width * n));
    for(size_t i = 0; i < n * width; i++)
    {
        array[i] = 0;
    }
    FloatChannelPtr ptr(new FloatChannel(n, width, array));
    addFloatChannel(ptr, name);
}

void PointBuffer2::createUCharChannel(UCharChannelPtr data, std::string name, size_t n, unsigned width)
{
    ucharArr array(new unsigned char(width * n));
    for(size_t i = 0; i < n * width; i++)
    {
        array[i] = 0;
    }
    UCharChannelPtr ptr(new UCharChannel(n, width, array));
    addUCharChannel(ptr, name);
}

void PointBuffer2::addFloatChannel(FloatChannelPtr data, std::string name)
{
    auto ret = m_floatChannels.insert(std::pair<std::string, FloatChannelPtr>(name, data));
    if(!ret.second )
    {
        std::cout << timestamp << "PointBuffer: Float channel '"
                  << name << "' already exist. Will not add data."
                  << std::endl;
    }
}

void PointBuffer2::addUCharChannel(UCharChannelPtr data, std::string name)
{
    auto ret = m_ucharChannels.insert(std::pair<std::string, UCharChannelPtr>(name, data));
    if(!ret.second)
    {
        std::cout << timestamp << "PointBuffer: UChar channel '"
                  << name << "' already exist. Will not add data."
                  << std::endl;
    }
}

bool PointBuffer2::hasUCharChannel(std::string name)
{
    auto it = m_ucharChannels.find(name);
    return !(it == m_ucharChannels.end());
}

bool PointBuffer2::hasFloatChannel(std::string name)
{
    auto it = m_floatChannels.find(name);
    return !(it == m_floatChannels.end());
}

unsigned PointBuffer2::ucharChannelWidth(std::string name)
{
    auto it = m_ucharChannels.find(name);
    if(it == m_ucharChannels.end())
    {
        return 0;
    }
    else
    {
        return it->second->width();
    }
}

unsigned PointBuffer2::floatChannelWidth(std::string name)
{
    auto it = m_floatChannels.find(name);
    if(it == m_floatChannels.end())
    {
        return 0;
    }
    else
    {
        return it->second->width();
    }
}

PointBuffer2::FloatProxy PointBuffer2::getFloatHandle(int idx, const std::string& name)
{
    auto it = m_floatChannels.find(name);
    if(it != m_floatChannels.end())
    {
        FloatChannelPtr ptr = it->second;
        if(ptr)
        {
            if(idx < ptr->n())
            {
                floatArr array = ptr->get();
                return FloatProxy(&array[idx], ptr->width());
            }
            else
            {
                std::cout << timestamp << "getFloatHandle(): Index " << idx
                          << " / " << ptr->n() << " out of bounds." << std::endl;
                return FloatProxy();
            }
        }
        else
        {
            std::cout << timestamp << "getFloatHandle(): Found nullptr." << std::endl;
            return FloatProxy();
        }
    }
    else
    {
        std::cout << timestamp << "getFloatHandle(): Could not find channel'"
                  << name << "'." << std::endl;
        return FloatProxy();
    }
}

PointBuffer2::UCharProxy PointBuffer2::getUCharHandle(int idx, const std::string& name)
{
    auto it = m_ucharChannels.find(name);
    if(it != m_ucharChannels.end())
    {
        UCharChannelPtr ptr = it->second;
        if(ptr)
        {
            if(idx < ptr->n())
            {
                ucharArr array = ptr->get();
                return UCharProxy(&array[idx], ptr->width());
            }
            else
            {
                std::cout << timestamp << "getUCharHandle(): Index " << idx
                          << " / " << ptr->n() << " out of bounds." << std::endl;
                return UCharProxy();
            }
        }
        else
        {
            std::cout << timestamp << "getUCharHandle(): Found nullptr." << std::endl;
            return UCharProxy();
        }
    }
    else
    {
        std::cout << timestamp << "getUCharHandle(): Could not find channel'"
                  << name << "'." << std::endl;
        return UCharProxy();
    }
}

PointBuffer2::FloatProxy PointBuffer2::point(int idx)
{
    if(idx < m_numPoints)
    {
        return FloatProxy(&(m_points->get())[idx], m_points->width());
    }
    else
    {
        std::cout << timestamp << "PointBuffer2::point(): Index " << idx
                  << " / " << m_numPoints << " out of bounds." << std::endl;
        return FloatProxy();
    }
}

PointBuffer2::FloatProxy PointBuffer2::normal(int idx)
{
    return getFloatHandle(idx, "normals");
}

}
