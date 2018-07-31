#include <lvr2/io/ChannelHandler.hpp>
#include <lvr2/io/Timestamp.hpp>

#include <iostream>

namespace lvr2
{

void ChannelHandler::addFloatChannel(floatArr data, std::string name, size_t n, unsigned width)
{
    FloatChannelPtr channel(new FloatChannel(n, width, data));
    addFloatChannel(channel, name);
}

void ChannelHandler::addUCharChannel(ucharArr data, std::string name, size_t n, unsigned width)
{
    UCharChannelPtr channel(new UCharChannel(n, width, data));
    addUCharChannel(channel, name);
}

void ChannelHandler::addEmptyFloatChannel(std::string name, size_t n, unsigned width)
{
    floatArr array(new float[width * n]);
    for(size_t i = 0; i < n * width; i++)
    {
        array[i] = 0;
    }
    FloatChannelPtr ptr(new FloatChannel(n, width, array));
    addFloatChannel(ptr, name);
}

void ChannelHandler::addEmptyUCharChannel(std::string name, size_t n, unsigned width)
{
    ucharArr array(new unsigned char[width * n]);
    for(size_t i = 0; i < n * width; i++)
    {
        array[i] = 0;
    }
    UCharChannelPtr ptr(new UCharChannel(n, width, array));
    addUCharChannel(ptr, name);
}

void ChannelHandler::addFloatChannel(FloatChannelPtr data, std::string name)
{
    auto ret = m_floatChannels.insert(std::pair<std::string, FloatChannelPtr>(name, data));
    if(!ret.second )
    {
        std::cout << timestamp << "ChannelHandler: Float channel '"
                  << name << "' already exist. Will not add data."
                  << std::endl;
    }
}

void ChannelHandler::addUCharChannel(UCharChannelPtr data, std::string name)
{
    auto ret = m_ucharChannels.insert(std::pair<std::string, UCharChannelPtr>(name, data));
    if(!ret.second)
    {
        std::cout << timestamp << "ChannelHandler: UChar channel '"
                  << name << "' already exist. Will not add data."
                  << std::endl;
    }
}

bool ChannelHandler::hasUCharChannel(std::string name)
{
    auto it = m_ucharChannels.find(name);
    return !(it == m_ucharChannels.end());
}

bool ChannelHandler::hasFloatChannel(std::string name)
{
    auto it = m_floatChannels.find(name);
    return !(it == m_floatChannels.end());
}

unsigned ChannelHandler::ucharChannelWidth(std::string name)
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

unsigned ChannelHandler::floatChannelWidth(std::string name)
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

ChannelHandler::FloatProxy ChannelHandler::getFloatHandle(int idx, const std::string& name)
{
    auto it = m_floatChannels.find(name);
    if(it != m_floatChannels.end())
    {
        FloatChannelPtr ptr = it->second;
        if(ptr)
        {
            if(idx < ptr->numAttributes())
            {
                floatArr array = ptr->dataPtr();
                return FloatProxy(&array[idx], ptr->width());
            }
            else
            {
                std::cout << timestamp << "ChannelHandler::getFloatHandle(): Index " << idx
                          << " / " << ptr->numAttributes() << " out of bounds." << std::endl;
                return FloatProxy();
            }
        }
        else
        {
            std::cout << timestamp << "ChannelHandler::getFloatHandle(): Found nullptr." << std::endl;
            return FloatProxy();
        }
    }
    else
    {
        std::cout << timestamp << "ChannelHandler::getFloatHandle(): Could not find channel'"
                  << name << "'." << std::endl;
        return FloatProxy();
    }
}



ChannelHandler::UCharProxy ChannelHandler::getUCharHandle(int idx, const std::string& name)
{
    auto it = m_ucharChannels.find(name);
    if(it != m_ucharChannels.end())
    {
        UCharChannelPtr ptr = it->second;
        if(ptr)
        {
            if(idx < ptr->numAttributes())
            {
                ucharArr array = ptr->dataPtr();
                return UCharProxy(&array[idx], ptr->width());
            }
            else
            {
                std::cout << timestamp << "ChannelHandler::getUCharHandle(): Index " << idx
                          << " / " << ptr->numAttributes() << " out of bounds." << std::endl;
                return UCharProxy();
            }
        }
        else
        {
            std::cout << timestamp << "ChannelHandler::getUCharHandle(): Found nullptr." << std::endl;
            return UCharProxy();
        }
    }
    else
    {
        std::cout << timestamp << "ChannelHandler::getUCharHandle(): Could not find channel'"
                  << name << "'." << std::endl;
        return UCharProxy();
    }
}

floatArr ChannelHandler::getFloatArray(size_t& n, unsigned& w, const std::string name)
{
    auto it = m_floatChannels.find(name);
    if(it != m_floatChannels.end())
    {
        n = it->second->numAttributes();
        w = it->second->width();
        return it->second->dataPtr();
    }
    else
    {
        n = 0;
        w = 0;
        return floatArr();
    }

}

ucharArr ChannelHandler::getUCharArray(size_t& n, unsigned& w, const std::string name)
{
    auto it = m_ucharChannels.find(name);
    if(it != m_ucharChannels.end())
    {
        n = it->second->numAttributes();
        w = it->second->width();
        return it->second->dataPtr();
    }
    else
    {
        n = 0;
        w = 0;
        return ucharArr();
    }
}

ChannelHandler::UCharChannel& ChannelHandler::getUCharChannel(std::string name)
{
    auto it = m_ucharChannels.find(name);
    if(it != m_ucharChannels.end())
    {
        return *(it->second);
    }
}

ChannelHandler::FloatChannel& ChannelHandler::getFloatChannel(std::string name)
{
    auto it = m_floatChannels.find(name);
    if(it != m_floatChannels.end())
    {
        return *(it->second);
    }

}

}
