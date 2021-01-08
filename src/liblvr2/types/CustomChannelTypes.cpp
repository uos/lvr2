#include "lvr2/types/CustomChannelTypes.hpp"
#include <cstring>

namespace lvr2 
{

template<>
boost::shared_array<unsigned char> serialize(
    const WaveformData& data, size_t& bsize)
{
    bsize = 0;
    bsize += sizeof(uint16_t);
    bsize += sizeof(bool);
    bsize += sizeof(uint16_t) * data.samples.size();

    boost::shared_array<unsigned char> ret(new unsigned char[bsize]);
    unsigned char* data_ptr = &ret[0];

    std::memcpy(data_ptr, reinterpret_cast<const unsigned char*>(&data.echo_type), sizeof(uint16_t));
    data_ptr += sizeof(uint16_t);
    std::memcpy(data_ptr, reinterpret_cast<const unsigned char*>(&data.low_power), sizeof(bool));
    data_ptr += sizeof(bool);
    std::memcpy(data_ptr, reinterpret_cast<const unsigned char*>(&data.samples[0]), sizeof(uint16_t) * data.samples.size());

    return ret;
}

template<>
boost::optional<WaveformData> deserialize(
    const unsigned char* buffer, const size_t& bsize)
{
    boost::optional<WaveformData> ret;
    WaveformData data;

    data.echo_type = *reinterpret_cast<const uint16_t*>(buffer);
    buffer += sizeof(uint16_t);
    data.low_power = *reinterpret_cast<const bool*>(buffer);
    buffer += sizeof(bool);
    size_t Nsamples = (bsize - sizeof(uint16_t) - sizeof(bool)) / (sizeof(uint16_t));
    data.samples.resize(Nsamples);
    unsigned char* data_ptr = reinterpret_cast<unsigned char*>(&data.samples[0]);
    std::memcpy(data_ptr, buffer, Nsamples * sizeof(uint16_t));

    ret = data;
    return ret;
}


} // namespace lvr2 