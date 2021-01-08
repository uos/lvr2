#pragma once

#ifndef LVR2_TYPES_CUSTOMCHANNELTYPES_HPP
#define LVR2_TYPES_CUSTOMCHANNELTYPES_HPP

#include <vector>
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
boost::shared_array<unsigned char> serialize(
    const T& data, size_t& bsize);

// default
template<typename T>
boost::shared_array<unsigned char> serialize(const T& data, size_t& bsize)
{
    boost::shared_array<unsigned char> ret;
    return ret;
}

template<typename T>
boost::optional<T> deserialize(const unsigned char* buffer, const size_t& bsize);

template<typename T>
boost::optional<T> deserialize(const unsigned char* buffer, const size_t& bsize)
{
    boost::optional<T> ret;
    return ret;
}



struct WaveformData {
    std::vector<uint16_t>   samples;
    uint16_t                echo_type;
    bool                    low_power;
};

template<>
boost::shared_array<unsigned char> serialize(
    const WaveformData& data, size_t& bsize);

template<>
boost::optional<WaveformData> deserialize(
    const unsigned char* buffer, const size_t& bsize);




} // namespace lvr2

#endif // LVR2_TYPES_CUSTOMCHANNELTYPES_HPP