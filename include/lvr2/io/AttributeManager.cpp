#include <lvr2/io/AttributeManager.hpp>
#include <lvr2/io/Timestamp.hpp>

#include <iostream>

namespace lvr2
{

void AttributeManager::addFloatChannel(floatArr data, std::string name, size_t n, unsigned width)
{
    FloatChannelPtr channel(new FloatChannel(n, width, data));
    addFloatChannel(channel, name);
}

void AttributeManager::addUCharChannel(ucharArr data, std::string name, size_t n, unsigned width)
{
    UCharChannelPtr channel(new UCharChannel(n, width, data));
    addUCharChannel(channel, name);
}

void AttributeManager::addIndexChannel(indexArray data, std::string name, size_t n, unsigned width)
{
    IndexChannelPtr channel(new IndexChannel(n, width, data));
    addIndexChannel(channel, name);
}

void AttributeManager::addEmptyFloatChannel(std::string name, size_t n, unsigned width)
{
    floatArr array(new float[width * n]);
    for(size_t i = 0; i < n * width; i++)
    {
        array[i] = 0;
    }
    FloatChannelPtr ptr(new FloatChannel(n, width, array));
    addFloatChannel(ptr, name);
}

void AttributeManager::addEmptyUCharChannel(std::string name, size_t n, unsigned width)
{
    ucharArr array(new unsigned char[width * n]);
    for(size_t i = 0; i < n * width; i++)
    {
        array[i] = 0;
    }
    UCharChannelPtr ptr(new UCharChannel(n, width, array));
    addUCharChannel(ptr, name);
}

void AttributeManager::addEmptyIndexChannel(std::string name, size_t n, unsigned width)
{
    indexArray array(new unsigned int[width * n]);
    for(size_t i = 0; i < n * width; i++)
    {
        array[i] = 0;
    }
    IndexChannelPtr ptr(new IndexChannel(n, width, array));
    addIndexChannel(ptr, name);
}

void AttributeManager::addFloatChannel(FloatChannelPtr data, std::string name)
{
    auto ret = m_floatChannels.insert(std::pair<std::string, FloatChannelPtr>(name, data));
    if(!ret.second )
    {
        std::cout << timestamp << "AttributeManager: Float channel '"
                  << name << "' already exists. Will not add data."
                  << std::endl;
    }
}

void AttributeManager::addUCharChannel(UCharChannelPtr data, std::string name)
{
    auto ret = m_ucharChannels.insert(std::pair<std::string, UCharChannelPtr>(name, data));
    if(!ret.second)
    {
        std::cout << timestamp << "AttributeManager: UChar channel '"
                  << name << "' already exists. Will not add data."
                  << std::endl;
    }
}

void AttributeManager::addIndexChannel(IndexChannelPtr data, std::string name)
{
    auto ret = m_indexChannels.insert(std::pair<std::string, IndexChannelPtr>(name, data));
    if(!ret.second)
    {
        std::cout << timestamp << "AttributeManager: Index channel '"
                  << name << "' already exists. Will not add data."
                  << std::endl;
    }
}

bool AttributeManager::hasUCharChannel(std::string name)
{
    auto it = m_ucharChannels.find(name);
    return !(it == m_ucharChannels.end());
}

bool AttributeManager::hasFloatChannel(std::string name)
{
    auto it = m_floatChannels.find(name);
    return !(it == m_floatChannels.end());
}

bool AttributeManager::hasIndexChannel(std::string name)
{
    auto it = m_indexChannels.find(name);
    return !(it == m_indexChannels.end());
}

unsigned AttributeManager::ucharChannelWidth(std::string name)
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

unsigned AttributeManager::floatChannelWidth(std::string name)
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

unsigned AttributeManager::indexChannelWidth(std::string name)
{
    auto it = m_indexChannels.find(name);
    if(it == m_indexChannels.end())
    {
        return 0;
    }
    else
    {
        return it->second->width();
    }
}

FloatProxy AttributeManager::getFloatHandle(int idx, const std::string& name)
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
                std::cout << timestamp << "AttributeManager::getFloatHandle(): Index " << idx
                          << " / " << ptr->numAttributes() << " out of bounds." << std::endl;
                return FloatProxy();
            }
        }
        else
        {
            std::cout << timestamp << "AttributeManager::getFloatHandle(): Found nullptr." << std::endl;
            return FloatProxy();
        }
    }
    else
    {
        std::cout << timestamp << "AttributeManager::getFloatHandle(): Could not find channel'"
                  << name << "'." << std::endl;
        return FloatProxy();
    }
}

UCharProxy AttributeManager::getUCharHandle(int idx, const std::string& name)
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
                std::cout << timestamp << "AttributeManager::getUCharHandle(): Index " << idx
                          << " / " << ptr->numAttributes() << " out of bounds." << std::endl;
                return UCharProxy();
            }
        }
        else
        {
            std::cout << timestamp << "AttributeManager::getUCharHandle(): Found nullptr." << std::endl;
            return UCharProxy();
        }
    }
    else
    {
        std::cout << timestamp << "AttributeManager::getUCharHandle(): Could not find channel'"
                  << name << "'." << std::endl;
        return UCharProxy();
    }
}

IndexProxy AttributeManager::getIndexHandle(int idx, const std::string& name)
{
    auto it = m_indexChannels.find(name);
    if(it != m_indexChannels.end())
    {
        IndexChannelPtr ptr = it->second;
        if(ptr)
        {
            if(idx < ptr->numAttributes())
            {
                indexArray array = ptr->dataPtr();
                return IndexProxy(&array[idx], ptr->width());
            }
            else
            {
                std::cout << timestamp << "AttributeManager::getIndexHandle(): Index " << idx
                          << " / " << ptr->numAttributes() << " out of bounds." << std::endl;
                return IndexProxy();
            }
        }
        else
        {
            std::cout << timestamp << "AttributeManager::getIndexHandle(): Found nullptr." << std::endl;
            return IndexProxy();
        }
    }
    else
    {
        std::cout << timestamp << "AttributeManager::getIndexHandle(): Could not find channel'"
                  << name << "'." << std::endl;
        return IndexProxy();
    }
}


floatArr AttributeManager::getFloatArray(size_t& n, unsigned& w, const std::string name)
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

ucharArr AttributeManager::getUCharArray(size_t& n, unsigned& w, const std::string name)
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

indexArray AttributeManager::getIndexArray(size_t& n, unsigned& w, const std::string name)
{
    auto it = m_indexChannels.find(name);
    if(it != m_indexChannels.end())
    {
        n = it->second->numAttributes();
        w = it->second->width();
        return it->second->dataPtr();
    }
    else
    {
        n = 0;
        w = 0;
        return indexArray();
    }
}

UCharChannelOptional AttributeManager::getUCharChannel(std::string name)
{
    auto it = m_ucharChannels.find(name);
    if(it != m_ucharChannels.end())
    {
        return UCharChannelOptional(*(it->second));
    }
    else
    {
        return boost::none;
    }
}

FloatChannelOptional AttributeManager::getFloatChannel(std::string name)
{
    auto it = m_floatChannels.find(name);
    if(it != m_floatChannels.end())
    {
        return FloatChannelOptional(*(it->second));
    }
    else
    {
        return boost::none;
    }

}


IndexChannelOptional AttributeManager::getIndexChannel(std::string name)
{
    auto it = m_indexChannels.find(name);
    if(it != m_indexChannels.end())
    {
        return IndexChannelOptional(*(it->second));
    }
    else
    {
        return boost::none;
    }
}

}
