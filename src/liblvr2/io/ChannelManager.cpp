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

#include "lvr2/io/ChannelManager.hpp"
#include "lvr2/io/Timestamp.hpp"

#include <iostream>

namespace lvr2
{

void ChannelManager::addFloatAtomic(float data, std::string name)
{
    auto ret = m_floatAtomics.insert(std::pair<std::string, float>(name, data));
    if(!ret.second )
    {
        std::cout << timestamp << "AtomicManager: Float Atomic '"
                  << name << "' already exists. Will not add data."
                  << std::endl;
    }
}

void ChannelManager::addUCharAtomic(unsigned char data, std::string name)
{
    auto ret = m_ucharAtomics.insert(std::pair<std::string, unsigned char>(name, data));
    if(!ret.second )
    {
        std::cout << timestamp << "AtomicManager: UChar Atomic '"
                  << name << "' already exists. Will not add data."
                  << std::endl;
    }
}


void ChannelManager::addIntAtomic(int data, std::string name)
{
    auto ret = m_intAtomics.insert(std::pair<std::string, int>(name, data));
    if(!ret.second )
    {
        std::cout << timestamp << "AtomicManager: Int Atomic '"
                  << name << "' already exists. Will not add data."
                  << std::endl;
    }
}

floatOptional ChannelManager::getFloatAtomic(std::string name)
{
    auto it = m_floatAtomics.find(name);
    if(it != m_floatAtomics.end())
    {
        return floatOptional(it->second);
    }
    else
    {
        return boost::none;
    }
}

ucharOptional ChannelManager::getUCharAtomic(std::string name)
{
    auto it = m_ucharAtomics.find(name);
    if(it != m_ucharAtomics.end())
    {
        return ucharOptional(it->second);
    }
    else
    {
        return boost::none;
    }
}

intOptional ChannelManager::getIntAtomic(std::string name)
{
    auto it = m_intAtomics.find(name);
    if(it != m_intAtomics.end())
    {
        return intOptional(it->second);
    }
    else
    {
        return boost::none;
    }
}

void ChannelManager::addFloatChannel(floatArr data, std::string name, size_t n, unsigned width)
{
    FloatChannelPtr channel(new FloatChannel(n, width, data));
    addFloatChannel(channel, name);
}

void ChannelManager::addUCharChannel(ucharArr data, std::string name, size_t n, unsigned width)
{
    UCharChannelPtr channel(new UCharChannel(n, width, data));
    addUCharChannel(channel, name);
}

void ChannelManager::addIndexChannel(indexArray data, std::string name, size_t n, unsigned width)
{
    IndexChannelPtr channel(new IndexChannel(n, width, data));
    addIndexChannel(channel, name);
}

void ChannelManager::addEmptyFloatChannel(std::string name, size_t n, unsigned width)
{
    floatArr array(new float[width * n]);
    for(size_t i = 0; i < n * width; i++)
    {
        array[i] = 0;
    }
    FloatChannelPtr ptr(new FloatChannel(n, width, array));
    addFloatChannel(ptr, name);
}

void ChannelManager::addEmptyUCharChannel(std::string name, size_t n, unsigned width)
{
    ucharArr array(new unsigned char[width * n]);
    for(size_t i = 0; i < n * width; i++)
    {
        array[i] = 0;
    }
    UCharChannelPtr ptr(new UCharChannel(n, width, array));
    addUCharChannel(ptr, name);
}

void ChannelManager::addEmptyIndexChannel(std::string name, size_t n, unsigned width)
{
    indexArray array(new unsigned int[width * n]);
    for(size_t i = 0; i < n * width; i++)
    {
        array[i] = 0;
    }
    IndexChannelPtr ptr(new IndexChannel(n, width, array));
    addIndexChannel(ptr, name);
}


void ChannelManager::addFloatChannel(FloatChannelPtr data, std::string name)
{
    auto ret = m_floatChannels.insert(std::pair<std::string, FloatChannelPtr>(name, data));
    if(!ret.second )
    {
        std::cout << timestamp << "AtomicManager: Float channel '"
                  << name << "' already exists. Will not add data."
                  << std::endl;
    }
}

void ChannelManager::addUCharChannel(UCharChannelPtr data, std::string name)
{
    auto ret = m_ucharChannels.insert(std::pair<std::string, UCharChannelPtr>(name, data));
    if(!ret.second)
    {
        std::cout << timestamp << "AtomicManager: UChar channel '"
                  << name << "' already exists. Will not add data."
                  << std::endl;
    }
}

void ChannelManager::addIndexChannel(IndexChannelPtr data, std::string name)
{
    auto ret = m_indexChannels.insert(std::pair<std::string, IndexChannelPtr>(name, data));
    if(!ret.second)
    {
        std::cout << timestamp << "AtomicManager: Index channel '"
                  << name << "' already exists. Will not add data."
                  << std::endl;
    }
}

bool ChannelManager::hasUCharChannel(std::string name)
{
    auto it = m_ucharChannels.find(name);
    return !(it == m_ucharChannels.end());
}

bool ChannelManager::hasFloatChannel(std::string name)
{
    auto it = m_floatChannels.find(name);
    return !(it == m_floatChannels.end());
}

bool ChannelManager::hasIndexChannel(std::string name)
{
    auto it = m_indexChannels.find(name);
    return !(it == m_indexChannels.end());
}

unsigned ChannelManager::ucharChannelWidth(std::string name)
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

unsigned ChannelManager::floatChannelWidth(std::string name)
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

unsigned ChannelManager::indexChannelWidth(std::string name)
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

FloatProxy ChannelManager::getFloatHandle(int idx, const std::string& name)
{
    auto it = m_floatChannels.find(name);
    if(it != m_floatChannels.end())
    {
        FloatChannelPtr ptr = it->second;
        if(ptr)
        {
            if(idx < ptr->numElements())
            {
                floatArr array = ptr->dataPtr();
                return FloatProxy(&array[idx], ptr->width());
            }
            else
            {
                std::cout << timestamp << "ChannelManager::getFloatHandle(): Index " << idx
                          << " / " << ptr->numElements() << " out of bounds." << std::endl;
                return FloatProxy();
            }
        }
        else
        {
            std::cout << timestamp << "ChannelManager::getFloatHandle(): Found nullptr." << std::endl;
            return FloatProxy();
        }
    }
    else
    {
        std::cout << timestamp << "ChannelManager::getFloatHandle(): Could not find channel'"
                  << name << "'." << std::endl;
        return FloatProxy();
    }
}

UCharProxy ChannelManager::getUCharHandle(int idx, const std::string& name)
{
    auto it = m_ucharChannels.find(name);
    if(it != m_ucharChannels.end())
    {
        UCharChannelPtr ptr = it->second;
        if(ptr)
        {
            if(idx < ptr->numElements())
            {
                ucharArr array = ptr->dataPtr();
                return UCharProxy(&array[idx], ptr->width());
            }
            else
            {
                std::cout << timestamp << "ChannelManager::getUCharHandle(): Index " << idx
                          << " / " << ptr->numElements() << " out of bounds." << std::endl;
                return UCharProxy();
            }
        }
        else
        {
            std::cout << timestamp << "ChannelManager::getUCharHandle(): Found nullptr." << std::endl;
            return UCharProxy();
        }
    }
    else
    {
        std::cout << timestamp << "ChannelManager::getUCharHandle(): Could not find channel'"
                  << name << "'." << std::endl;
        return UCharProxy();
    }
}

IndexProxy ChannelManager::getIndexHandle(int idx, const std::string& name)
{
    auto it = m_indexChannels.find(name);
    if(it != m_indexChannels.end())
    {
        IndexChannelPtr ptr = it->second;
        if(ptr)
        {
            if(idx < ptr->numElements())
            {
                indexArray array = ptr->dataPtr();
                return IndexProxy(&array[idx], ptr->width());
            }
            else
            {
                std::cout << timestamp << "ChannelManager::getIndexHandle(): Index " << idx
                          << " / " << ptr->numElements() << " out of bounds." << std::endl;
                return IndexProxy();
            }
        }
        else
        {
            std::cout << timestamp << "ChannelManager::getIndexHandle(): Found nullptr." << std::endl;
            return IndexProxy();
        }
    }
    else
    {
        std::cout << timestamp << "ChannelManager::getIndexHandle(): Could not find channel'"
                  << name << "'." << std::endl;
        return IndexProxy();
    }
}


floatArr ChannelManager::getFloatArray(size_t& n, unsigned& w, const std::string name)
{
    auto it = m_floatChannels.find(name);
    if(it != m_floatChannels.end())
    {
        n = it->second->numElements();
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

ucharArr ChannelManager::getUCharArray(size_t& n, unsigned& w, const std::string name)
{
    auto it = m_ucharChannels.find(name);
    if(it != m_ucharChannels.end())
    {
        n = it->second->numElements();
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

indexArray ChannelManager::getIndexArray(size_t& n, unsigned& w, const std::string name)
{
    auto it = m_indexChannels.find(name);
    if(it != m_indexChannels.end())
    {
        n = it->second->numElements();
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

UCharChannelOptional ChannelManager::getUCharChannel(std::string name)
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

FloatChannelOptional ChannelManager::getFloatChannel(std::string name)
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


IndexChannelOptional ChannelManager::getIndexChannel(std::string name)
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
