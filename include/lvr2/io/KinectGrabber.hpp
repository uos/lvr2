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
 * KinecGrabber.h
 *
 *  Created on: 20.03.2012
 *      Author: Thomas Wiemann
 */

#ifndef KINECGRABBER_H_
#define KINECGRABBER_H_

#include "lvr2/io/PointBuffer.hpp"
#include "libfreenect.hpp"

#include <boost/thread.hpp"
#include <vector>

namespace lvr2
{

class KinectGrabber : public Freenect::FreenectDevice
{
public:
	KinectGrabber(freenect_context *_ctx, int _index);
	virtual ~KinectGrabber();

	/// Returns the currently present point cloud data
	void getDepthImage(std::vector<short> &img);
	void getColorImage(std::vector<uint8_t> &img);

protected:
	virtual void VideoCallback(void* data, uint32_t timestamp);
	virtual void DepthCallback(void* data, uint32_t timestamp);

	/// PointBufferPtr with current data
	PointBufferPtr			m_buffer;

	/// Mutex for save depth buffer access
	boost::mutex			m_depthMutex;

	/// Mutex for save color buffer access
	boost::mutex			m_colorMutex;

	/// The raw depth image
	std::vector<short>		m_depthImage;

	/// The raw color image
	std::vector<uint8_t>	m_colorImage;

	bool					m_haveData;

};

} // namespace lvr2

#endif /* KINECGRABBER_H_ */
