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
 * SignalingKinectGrabber.cpp
 *
 *  Created on: 02.04.2012
 *      Author: Thomas Wiemann
 */

#include "SignalingKinectGrabber.hpp"
#include "io/DataStruct.hpp"

#include <iostream>
using namespace std;


SignalingKinectGrabber::SignalingKinectGrabber(freenect_context *_ctx, int _index)
	: KinectGrabber(_ctx, _index)
{
	// Dept calibration matrix initialization
	float fx = 594.21f;
	float fy = 591.04f;
	float a = -0.0030711f;
	float b = 3.3309495f;
	float cx = 339.5f;
	float cy = 242.7f;

	m_depthMatrix <<
			1/fx,     0,  0, 0,
			0,    -1/fy,  0, 0,
			0,       0,  0, a,
			-cx/fx, cy/fy, -1, b;

	m_buffer = PointBufferPtr(new PointBuffer);
}

SignalingKinectGrabber::~SignalingKinectGrabber()
{
	// TODO Auto-generated destructor stub
}

void SignalingKinectGrabber::VideoCallback(void* data, uint32_t timestamp)
{
	KinectGrabber::VideoCallback(data, timestamp);
}

void SignalingKinectGrabber::DepthCallback(void* data, uint32_t timestamp)
{
	KinectGrabber::DepthCallback(data, timestamp);
	using namespace lssr;
	floatArr points(new float[m_depthImage.size() * 3]);

	int i,j;
	int c = 0;
	for (i = 0; i < 480; i++) {
		for (j = 0; j < 640; j++) {

			Eigen::Vector4f v;
			v << j, i, (float)(m_depthImage[i * 640 + j]), 1.0f;
			v = m_depthMatrix.transpose() * v;

			points[3 * c    ] = v(0) / v(3);
			points[3 * c + 1] = v(1) / v(3);
			points[3 * c + 2] = v(2) / v(3);
			c++;
			//cout << points[3 * c] << " " << points[3 * c + 1] << " " << points[3 * c + 2] << endl;
		}
	}
	m_buffer->setPointArray(points, 640 * 480);
	Q_EMIT newPointBuffer(&m_buffer);
	usleep(100000);
}

