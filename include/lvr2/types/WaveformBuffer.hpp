#ifndef WAVEFORMBUFFER_HPP
#define WAVEFORMBUFFER_HPP

#include "lvr2/types/PointBuffer.hpp"

#include <map>
#include <string>

#include <boost/shared_array.hpp>
#include <iostream>

namespace lvr2
{

class WaveformBuffer : public PointBuffer
{
    using base = BaseBuffer;
public:    
    WaveformBuffer() : PointBuffer(){};
    WaveformBuffer(floatArr points, size_t n) : PointBuffer(points, n){};
    WaveformBuffer(floatArr points, floatArr normals, size_t n) : PointBuffer(points, normals, n){};

    /***
     * @brief Adds points to the buffer. If the buffer already
     *        contains point cloud data, the interal buffer will
     *        be freed als well as all other attribute channels.
     */
    void setWaveformArray(boost::shared_array<uint16_t> waveformData, boost::shared_array<size_t> waveformSize)
    {
	m_waveform = waveformData;
        m_waveformSize = waveformSize;
    }

    boost::shared_array<uint16_t> getWaveformArray()
    {
        return m_waveform;
    }
    boost::shared_array<size_t> getWaveformSize()
    {
        return m_waveformSize;
    }

    /// Makes a clone
    WaveformBuffer clone();

private:
    boost::shared_array<uint16_t> m_waveform;
    boost::shared_array<size_t> m_waveformSize;
};

using WaveformBufferPtr = std::shared_ptr<WaveformBuffer>;

} // namespace lvr2

#endif // POINTBUFFER2_HPP
