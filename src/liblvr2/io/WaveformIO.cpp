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

 /*
 * WaveformIO.cpp
 *
 *  Created on: 23.06.2020
 *      Author: Thomas Wiemann
 */

#include <fstream>
#include <string.h>
#include <algorithm>
#include <ios>
#include <array>

using std::ifstream;

#include <boost/filesystem.hpp>

#include "lvr2/io/WaveformIO.hpp"
#include "lvr2/io/Progress.hpp"
#include "lvr2/io/Timestamp.hpp"

namespace lvr2
{

ModelPtr WaveformIO::read(std::string filename)
{
    boost::filesystem::path selectedFile(filename);
    ModelPtr model(new Model);
    
    // Open file
    ifstream in(filename.c_str(), std::ios::binary);
    //in.open();

    model->m_pointCloud = PointBufferPtr( new PointBuffer);

    std::vector<std::array<float,3>> points;

    char floatBuffer[sizeof(float)];
    char uint16Buffer[sizeof(uint16_t)];
    char uint32Buffer[sizeof(uint32_t)];
    float x, y, z, time;
    uint32_t sampleCount;
    uint16_t sampleValue;
    int counter = 0;
    int maxSampleCount = 0;
    std::vector<std::vector<uint16_t>> waveforms;
    while(in.good())
    {
	//read x,y,z
    	in.read(floatBuffer, sizeof(float));
    	x = *(float *)floatBuffer;
    	in.read(floatBuffer, sizeof(float));
    	y = *(float *)floatBuffer;
    	in.read(floatBuffer, sizeof(float));
    	z = *(float *)floatBuffer;

	std::array<float, 3> tmp = {x,y,z};
	points.push_back(tmp);
        
	//get Time
	in.read(floatBuffer, sizeof(float));
    	time = *(float *)floatBuffer;

	//read waveforms
	//get sample count of waveform
    	in.read(uint32Buffer, sizeof(uint32_t));
    	sampleCount = *(uint32_t *)uint32Buffer;
        std::vector<uint16_t> waveform;
        waveform.resize(sampleCount);
        if(sampleCount > maxSampleCount)
        {
            maxSampleCount = sampleCount;
        }
	for (int i = 0; i < sampleCount; i++)
	{
    	    in.read(uint16Buffer, sizeof(uint16_t));
	    waveform[i] = *(uint16_t *)uint16Buffer;
	}
	waveforms.push_back(waveform);
	counter++;
    }

    floatArr outPoints;
    outPoints = floatArr( new float[ (points.size() -1) * 3 ] );
    for (int i = 0; i < points.size() -1 ; i++)
    {
        outPoints[3 * i] = points[i][0];
        outPoints[(3 * i) + 1] = points[i][1];
        outPoints[(3 * i) + 2] = points[i][2];

    }

    //TODO A better way is needed
    boost::shared_array<uint16_t> waveformData(new uint16_t[maxSampleCount * (points.size() - 1)]);
    for (int i = 0; i < points.size() -1 ; i++)
    {
	for (int j = 0; j < maxSampleCount; j++)
	{
	    if (j < waveforms[i].size() && waveforms[i].size() > 0)
	    {
        	waveformData[(maxSampleCount * i) + j] = waveforms[i][j];
	    }
	    else
	    {
        	waveformData[(maxSampleCount * i) + j] = 0;
	    }
	}

    }
    Channel<uint16_t>::Ptr waveformChannel(new Channel<uint16_t>(points.size() - 1, maxSampleCount, waveformData));
    

    PointBufferPtr pc;
    pc = PointBufferPtr( new PointBuffer );
    pc->setPointArray(outPoints, (points.size() -1));
    pc->addChannel<uint16_t>(waveformChannel, "waveform");
    ModelPtr outModel(new Model(pc));
    this->m_model = outModel;
    return outModel;
}



void WaveformIO::save( std::string filename )
{
}



} // namespace lvr
