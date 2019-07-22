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

    ElementProxy<T> operator[](const unsigned& idx);

    size_t     width() const;
    size_t     numElements() const;
    DataPtr    dataPtr() const;

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

using UCharChannel = Channel<unsigned char>;
using UCharChannelOptional = UCharChannel::Optional;
using UCharChannelPtr = UCharChannel::Ptr;

using IndexChannel = Channel<unsigned int>;
using IndexChannelOptional = IndexChannel::Optional;
using IndexChannelPtr = IndexChannel::Ptr;


} // namespace lvr2

#include "Channel.tcc"

#endif // LVR2_TYPES_CHANNEL