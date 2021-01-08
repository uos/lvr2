#include "lvr2/types/CustomChannelTypes.hpp"

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
    
    

    return ret;
}

template<>
boost::optional<WaveformData> deserialize(
    const unsigned char* buffer, const size_t& bsize)
{
    boost::optional<WaveformData> ret;
    WaveformData data;

    ret = data;
    return ret;
}


} // namespace lvr2 