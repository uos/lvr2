#include "lvr2/io/WaveformBuffer.hpp"
#include "lvr2/io/Timestamp.hpp"

#include <iostream>

namespace lvr2
{

WaveformBuffer WaveformBuffer::clone() 
{
    WaveformBuffer wfb;

    floatArr pb = PointBuffer::getPointArray();
    wfb.setPointArray(pb, numPoints());
    if (hasNormals())
    {
        wfb.setNormalArray(getNormalArray(), numPoints());
    }
    
    if (hasColors())
    {
        size_t colorArraySize;
        ucharArr colorArray = PointBuffer::getColorArray(colorArraySize);
        wfb.setColorArray(colorArray, numPoints(), colorArraySize);
    }

    wfb.setWaveformArray(m_waveform, m_waveformSize);
    return wfb;
}
}


