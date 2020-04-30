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

#pragma once

#ifndef LVR2_TYPES_CHANNEL
#define LVR2_TYPES_CHANNEL

#include "ElementProxy.hpp"
#include <memory>
#include <boost/optional.hpp>
#include <boost/shared_array.hpp>
#include <iostream>

namespace lvr2 {

template<typename T>
class Channel
{
public:
    using Optional =  boost::optional<Channel<T> >;
    using Ptr = std::shared_ptr<Channel<T> >;
    using DataType = T;
    using DataPtr = boost::shared_array<T>;

    Channel();
    Channel(size_t n, size_t width);
    Channel(size_t n, size_t width, DataPtr ptr);
    

    // clone
    Channel<T> clone() const;

    ElementProxy<T> operator[](const unsigned& idx);
    const ElementProxy<T> operator[](const unsigned& idx) const;

    size_t           width() const;
    size_t           numElements() const;
    const DataPtr    dataPtr() const;
    DataPtr          dataPtr();

    friend std::ostream& operator<<(std::ostream& os, const Channel<T>& ch)
    {
        os << "size: [" << ch.numElements() << "," << ch.width() << "]";
        return os;
    }

protected:
    size_t          m_numElements;
    size_t          m_elementWidth;
    DataPtr         m_data;
};

template<typename T>
using AttributeChannel = Channel<T>;

template<typename T>
using ChannelPtr = typename Channel<T>::Ptr;

template<typename T>
using ChannelOptional = typename Channel<T>::Optional;

using FloatChannel = Channel<float>;
using FloatChannelOptional = FloatChannel::Optional;
using FloatChannelPtr = FloatChannel::Ptr;

using DoubleChannel = Channel<double>;
using DoubleChannelOptional = DoubleChannel::Optional;
using DoubleChannelPtr = DoubleChannel::Ptr;

using UCharChannel = Channel<unsigned char>;
using UCharChannelOptional = UCharChannel::Optional;
using UCharChannelPtr = UCharChannel::Ptr;

using IndexChannel = Channel<unsigned int>;
using IndexChannelOptional = IndexChannel::Optional;
using IndexChannelPtr = IndexChannel::Ptr;


} // namespace lvr2

#include "Channel.tcc"

#endif // LVR2_TYPES_CHANNEL