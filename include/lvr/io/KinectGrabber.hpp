/* Copyright (C) 2011 Uni Osnabrück
 * This file is part of the LAS VEGAS Reconstruction Toolkit,
 *
 * LAS VEGAS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * LAS VEGAS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA
 */

/*
 * KinecGrabber.h
 *
 *  Created on: 20.03.2012
 *      Author: Thomas Wiemann
 */

#ifndef KINECGRABBER_H_
#define KINECGRABBER_H_

#include "io/PointBuffer.hpp"
#include "libfreenect.hpp"

#include <boost/thread.hpp>
#include <vector>

namespace lvr
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

}

#endif /* KINECGRABBER_H_ */
