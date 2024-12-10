#ifndef LVR2_TYPES_BYTE_ENCODING_HPP
#define LVR2_TYPES_BYTE_ENCODING_HPP

#include <boost/shared_array.hpp>
#include <boost/optional.hpp>

namespace lvr2 {

/**
 * @brief Specialize this method in CustomChannelTypes.cpp for your CustomType.
 *        Only if specialized, datatype can be stored.
 * 
 * @tparam T 
 * @param data 
 * @param bsize 
 * @return boost::shared_array<unsigned char> 
 */
template<typename T>
boost::shared_array<unsigned char> byteEncode(
    const T& data, size_t& bsize);

// default
template<typename T>
boost::shared_array<unsigned char> byteEncode(const T& data, size_t& bsize)
{
    boost::shared_array<unsigned char> ret;
    return ret;
}

template<typename T>
boost::optional<T> byteDecode(const unsigned char* buffer, const size_t& bsize);

template<typename T>
boost::optional<T> byteDecode(const unsigned char* buffer, const size_t& bsize)
{
    boost::optional<T> ret;
    return ret;
}

} // namespace lvr2

#endif // LVR2_TYPES_BYTE_ENCODING_HPP