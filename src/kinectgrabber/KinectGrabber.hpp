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

#include <pcl/io/openni_grabber.h>
#include <pcl/point_types.h>
#include <boost/thread.hpp>

#include "io/PointBuffer.hpp"

namespace lssr
{

class KinectGrabber
{
public:
	KinectGrabber(bool autostart = false);
	virtual ~KinectGrabber();

	/// Starts grabbing data
	void start();

	/// Stops grabbing data
	void stop();

	/// Returns the currently present point cloud data
	PointBufferPtr getBuffer();

private:

	/// Kinect update callback
	void update(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud);

	/// Mutex for save data access
	boost::mutex 	m_mutex;

	/// PointBufferPtr with current data
	PointBufferPtr	m_buffer;

	/// PCL OpenNI grabber
	pcl::Grabber*	m_grabber;

};

}

#endif /* KINECGRABBER_H_ */
