/*
 * KinecGrabber.cpp
 *
 *  Created on: 20.03.2012
 *      Author: Thomas Wiemann
 */

#include <iostream>
using std::cout;
using std::endl;

#include "KinectGrabber.hpp"
#include "io/Timestamp.hpp"



namespace lssr
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

} // namespace lssr
