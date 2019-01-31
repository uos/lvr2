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

#ifndef CHANNELHANDLER_HPP
#define CHANNELHANDLER_HPP

#include <lvr2/io/DataStruct.hpp>
#include <iostream>
#include <boost/optional.hpp>

namespace lvr2
{

template<typename T>
class ElementProxy
{
public:
    template<typename BaseVecT>
    ElementProxy operator=(const BaseVecT& v)
    {
        if( m_ptr && (m_w > 2))
        {
            m_ptr[0] = v.x;
            m_ptr[1] = v.y;
            m_ptr[2] = v.z;
        }
        return *this;
    }

    template<typename BaseVecT>
    ElementProxy operator+=(const BaseVecT& v)
    {
        if( m_ptr && (m_w > 2))
        {
            m_ptr[0] += v.x;
            m_ptr[1] += v.y;
            m_ptr[2] += v.z;
        }
        return *this;
    }

    template<typename BaseVecT>
    ElementProxy operator-=(const BaseVecT& v)
    {
        if( m_ptr && (m_w > 2))
        {
            m_ptr[0] -= v.x;
            m_ptr[1] -= v.y;
            m_ptr[2] -= v.z;
        }
        return *this;
    }


    template<typename BaseVecT>
    BaseVecT operator+(const BaseVecT& v)
    {
        if(m_w > 2)
        {
            *this += v;
            return BaseVecT(m_ptr[0], m_ptr[1], m_ptr[2]);
        }
        else
        {
            return BaseVecT(0, 0, 0);
        }
    }

    template<typename BaseVecT>
    BaseVecT operator-(const BaseVecT& v)
    {
        if(m_w > 2)
        {
            *this -= v;
            return BaseVecT(m_ptr[0], m_ptr[1], m_ptr[2]);
        }
        else
        {
            return BaseVecT(0, 0, 0);
        }
    }

    ElementProxy(T* pos = nullptr, unsigned w = 0) : m_ptr(pos), m_w(w) {}

    T operator[](int i) const
    {
        if(m_ptr && (i < m_w))
        {
            return m_ptr[i];
        }
        else
        {
            return 0;
        }
    }

    /// User defined conversion operator
    template<typename BaseVecT>
    operator BaseVecT() const
    {
        if(m_w == 3)
        {
            return BaseVecT(m_ptr[0], m_ptr[1], m_ptr[2]);
        }
        return BaseVecT(0, 0, 0);
    }

private:

    T*              m_ptr;
    unsigned        m_w;
};

template<typename T>
class AttributeChannel
{
public:
    using DataPtr = boost::shared_array<T>;

    AttributeChannel(size_t n, unsigned width)
        : m_width(width), m_numAttributes(n)
    {
        m_data = DataPtr(new T[m_numAttributes * width]);
    }

    AttributeChannel(size_t n, unsigned width, DataPtr ptr)
        : m_numAttributes(n),
          m_width(width),
          m_data(ptr)
    {

    }

    ElementProxy<T> operator[](const unsigned& idx)
    {
        T* ptr = m_data.get();
        return ElementProxy<T>(&(ptr[idx * m_width]), m_width);
    }

    DataPtr&     dataPtr() { return m_data;}
    unsigned     width() const { return m_width;}
    size_t       numAttributes() const { return m_numAttributes;}

private:
    size_t          m_numAttributes;
    unsigned        m_width;
    DataPtr         m_data;
};

// Some type aliases
using FloatChannel = AttributeChannel<float>;
using UCharChannel = AttributeChannel<unsigned char>;
using IndexChannel = AttributeChannel<unsigned int>;

using FloatChannelOptional = boost::optional<FloatChannel&>;
using UCharChannelOptional= boost::optional<UCharChannel&>;
using IndexChannelOptional = boost::optional<IndexChannel&>;

using FloatProxy = ElementProxy<float>;
using UCharProxy = ElementProxy<unsigned char>;
using IndexProxy = ElementProxy<unsigned int>;

using FloatChannelPtr = std::shared_ptr<FloatChannel>;
using UCharChannelPtr = std::shared_ptr<UCharChannel>;
using IndexChannelPtr = std::shared_ptr<IndexChannel>;

using intOptional = boost::optional<int>;
using floatOptional = boost::optional<float>;
using ucharOptional = boost::optional<unsigned char>;

class AttributeManager
{
public:
    AttributeManager() {}

    void addIndexChannel(
            indexArray array,
            std::string name,
            size_t n,
            unsigned width);


    void addFloatChannel(
            floatArr array,
            std::string name,
            size_t n,
            unsigned width);

    void addUCharChannel(
            ucharArr array,
            std::string name,
            size_t n,
            unsigned width);

    void addEmptyFloatChannel(
            std::string name,
            size_t n,
            unsigned width);

    void addEmptyUCharChannel(
            std::string name,
            size_t n,
            unsigned width);

    void addEmptyIndexChannel(
            std::string name,
            size_t n,
            unsigned width);

    void addChannel(indexArray array, std::string name, size_t n, unsigned width)
        { addIndexChannel(array, name, n, width);}

    void addChannel(floatArr array, std::string name, size_t n, unsigned width)
        { addFloatChannel(array, name, n, width);}

    void addChannel(ucharArr array, std::string name, size_t n, unsigned width)
        { addUCharChannel(array, name, n, width);}

    void getChannel(std::string name, FloatChannelOptional& channelOptional )
        { channelOptional = getFloatChannel(name);}

    void getChannel(std::string name, IndexChannelOptional& channelOptional)
        { channelOptional = getIndexChannel(name);}

    void getChannel(std::string name, UCharChannelOptional& channelOptional)
        { channelOptional = getUCharChannel(name);}

    bool hasUCharChannel(std::string name);
    bool hasFloatChannel(std::string name);
    bool hasIndexChannel(std::string name);

    unsigned ucharChannelWidth(std::string name);
    unsigned floatChannelWidth(std::string name);
    unsigned indexChannelWidth(std::string name);

    FloatProxy getFloatHandle(int idx, const std::string& name);
    UCharProxy getUCharHandle(int idx, const std::string& name);
    IndexProxy getIndexHandle(int idx, const std::string& name);

    FloatProxy operator[](size_t idx);

    floatArr getFloatArray(size_t& n, unsigned& w, const std::string name);
    ucharArr getUCharArray(size_t& n, unsigned& w, const std::string name);
    indexArray getIndexArray(size_t& n, unsigned& w, const std::string name);

    FloatChannelOptional getFloatChannel(std::string name);
    UCharChannelOptional getUCharChannel(std::string name);
    IndexChannelOptional getIndexChannel(std::string name);

    floatOptional getFloatAttribute(std::string name);
    ucharOptional getUCharAttribute(std::string name);
    intOptional getIntAttribute(std::string name);

    void addFloatChannel(FloatChannelPtr data, std::string name);
    void addUCharChannel(UCharChannelPtr data, std::string name);
    void addIndexChannel(IndexChannelPtr data, std::string name);

    void addFloatAttribute(float data, std::string name);
    void addUCharAttribute(unsigned char data, std::string name);
    void addIntAttribute(int data, std::string name);

private:

    std::map<std::string, FloatChannelPtr>      m_floatChannels;
    std::map<std::string, UCharChannelPtr>      m_ucharChannels;
    std::map<std::string, IndexChannelPtr>      m_indexChannels;

    std::map<std::string, float>                m_floatAttributes;
    std::map<std::string, unsigned char>        m_ucharAttributes;
    std::map<std::string, int>                  m_intAttributes;


    using FloatChannelMap = std::map<std::string, FloatChannelPtr>;
    using UCharChannelMap = std::map<std::string, UCharChannelPtr>;
    using IndexChannelMap = std::map<std::string, IndexChannelPtr>;
};

} // namespace lvr2

#endif // CHANNELHANDLER_HPP
