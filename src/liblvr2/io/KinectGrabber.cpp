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
 * KinecGrabber.cpp
 *
 *  Created on: 20.03.2012
 *      Author: Thomas Wiemann
 */

#include <iostream>
using std::cout;
using std::endl;

#include "lvr2/io/KinectGrabber.hpp"
#include "lvr2/io/Timestamp.hpp"

namespace lvr2
{

KinectGrabber::KinectGrabber(freenect_context *_ctx, int _index)
	: Freenect::FreenectDevice(_ctx, _index),
	m_haveData(false)
{
	m_depthImage = std::vector<short>( 640 * 480, 0 );
	m_colorImage = std::vector<uint8_t>(freenect_find_video_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_VIDEO_RGB).bytes,0 );

}

void KinectGrabber::VideoCallback(void* data, uint32_t timestamp)
{
	m_colorMutex.lock();
	uint8_t* rgb = static_cast<uint8_t*>(data);
	std::copy(rgb, rgb+getVideoBufferSize(), m_colorImage.begin());
	m_colorMutex.unlock();
}

/// Returns the currently present point cloud data
void KinectGrabber::getDepthImage(std::vector<short> &img)
{
	if(m_haveData)
	{
		m_depthMutex.lock();
		img.swap(m_depthImage);
		m_depthMutex.unlock();
		m_haveData = false;
	}
	else
	{
		img.clear();
		img.resize(0);
	}
}

void KinectGrabber::getColorImage(std::vector<uint8_t> &img)
{
	m_colorMutex.lock();
	img.swap(m_colorImage);
	m_colorMutex.unlock();
}


void KinectGrabber::DepthCallback(void* data, uint32_t timestamp)
{
	m_depthMutex.lock();
	short* depth = static_cast<short*>(data);
	for( unsigned int i = 0 ; i < 640*480 ; i++) {
		m_depthImage[i] = depth[i];
	}
	m_haveData = true;
	m_depthMutex.unlock();
}


KinectGrabber::~KinectGrabber()
{


}

} // namespace lvr2
