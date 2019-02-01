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

#include <lvr2/io/BaseBuffer.hpp>

namespace lvr2
{

BaseBuffer::BaseBuffer()
{

}

floatArr BaseBuffer::getFloatArray(const std::string& name, size_t& n, unsigned& w)
{
    return m_channels.getFloatArray(n, w, name);
}

ucharArr BaseBuffer::getUCharArray(const std::string& name, size_t& n, unsigned& w)
{
    return m_channels.getUCharArray(n, w, name);
}

indexArray BaseBuffer::getIndexArray(const std::string& name, size_t& n, unsigned& w)
{
    return m_channels.getIndexArray(n, w, name);
}

FloatChannelOptional BaseBuffer::getFloatChannel(const std::string& name)
{
    return m_channels.getFloatChannel(name);
}

UCharChannelOptional BaseBuffer::getUCharChannel(const std::string& name)
{
    return m_channels.getUCharChannel(name);
}

IndexChannelOptional BaseBuffer::getIndexChannel(const std::string& name)
{
    return m_channels.getIndexChannel(name);
}

floatOptional BaseBuffer::getFloatAttribute(std::string name)
{
    return m_channels.getFloatAttribute(name);
}

ucharOptional BaseBuffer::getUCharAttribute(std::string name)
{
    return m_channels.getUCharAttribute(name);
}

intOptional BaseBuffer::getIntAttribute(std::string name)
{
    return m_channels.getIntAttribute(name);
}


void BaseBuffer::addFloatChannel(floatArr data, std::string name, size_t n, unsigned w)
{
    m_channels.addFloatChannel(data, name, n, w);
}


void BaseBuffer::addUCharChannel(ucharArr data, std::string name, size_t n, unsigned w)
{
    m_channels.addUCharChannel(data, name, n, w);
}

void BaseBuffer::addIndexChannel(indexArray data, std::string name, size_t n, unsigned w)
{
    m_channels.addIndexChannel(data, name, n, w);
}

void BaseBuffer::addFloatAttribute(float data, std::string name)
{
    m_channels.addFloatAttribute(data, name);
}

void BaseBuffer::addUCharAttribute(unsigned char data, std::string name)
{
    m_channels.addUCharAttribute(data, name);
}

void BaseBuffer::addIntAttribute(int data, std::string name)
{
    m_channels.addIntAttribute(data, name);
}

bool BaseBuffer::addAttributeMap(const DenseAttrMap<BaseHandle<Index>, BaseVector<float> >& map, const std::string& name)
{
    floatArr values(new float[map.numValues() * 3]);
    Index i = 0;
    for(auto handle: map){
        values[i*3+0] = map[handle].x;
        values[i*3+1] = map[handle].y;
        values[i*3+2] = map[handle].z;
        i++;
    }
    m_channels.addFloatChannel(values, name, map.numValues(), 3);
    return true;
}

bool BaseBuffer::addAttributeMap(const AttributeMap<BaseHandle<Index>, BaseVector<float> >& map,
                     const std::string& keys_name, const std::string& values_name)
{
    floatArr values(new float[map.numValues() * 3]);
    indexArray keys(new unsigned int[map.numValues()]);
    Index i = 0;
    for(auto handle: map){
        values[i*3+0] = map[handle].x;
        values[i*3+1] = map[handle].y;
        values[i*3+2] = map[handle].z;
        keys[i++] = handle.idx();
    }
    m_channels.addFloatChannel(values, values_name, map.numValues(), 3);
    m_channels.addIndexChannel(keys, keys_name, map.numValues(), 1);
    return true;
}

bool BaseBuffer::getAttributeMap(DenseAttrMap <BaseHandle<Index>, BaseVector<float> >& map, const std::string& name)
{
  FloatChannelOptional values_opt = m_channels.getFloatChannel(name);

  if(values_opt && values_opt.get().width() == 3)
  {
    FloatChannel& values = values_opt.get();
    map.clear();
    for(size_t i=0; i<values.numAttributes(); i++)
    {
      map.insert(BaseHandle<Index>(i), values[i]);
    }
    return true;
  }
  else return false;
}

bool BaseBuffer::getAttributeMap(AttributeMap<BaseHandle<Index>, BaseVector<float> >& map,
                     const std::string& keys_name, const std::string& values_name)
{
    IndexChannelOptional keys_opt = m_channels.getIndexChannel(keys_name);
    FloatChannelOptional values_opt = m_channels.getFloatChannel(values_name);

    if(keys_opt && values_opt &&
        keys_opt.get().width() == 1 &&
        values_opt.get().width() == 3 &&
        keys_opt.get().numAttributes() == values_opt.get().numAttributes())
    {
        FloatChannel& values = values_opt.get();
        IndexChannel& keys = keys_opt.get();
        map.clear();
        for(size_t i=0; i<keys.numAttributes(); i++)
        {
            map.insert(BaseHandle<Index>(keys[i][0]), values[i]);
        }
        return true;
    }
    else return false;
}

}


