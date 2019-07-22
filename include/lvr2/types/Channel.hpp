#pragma once

#include "lvr2/io/ChannelManager.hpp"
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


} // namespace lvr2

#include "Channel.tcc"