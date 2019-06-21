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
 * KinectIO.cpp
 *
 *  Created on: 20.03.2012
 *      Author: Thomas Wiemann
 */

#include "lvr2/io/KinectIO.hpp"
#include "lvr2/io/PointBuffer.hpp"
#include "lvr2/io/DataStruct.hpp"

#include <Eigen/Dense>

#include <vector>
#include <set>

namespace lvr2
{

KinectIO* KinectIO::m_instance = 0;

KinectIO* KinectIO::instance()
{
	if(KinectIO::m_instance == 0)
	{
		KinectIO::m_instance = new KinectIO;
	}

	return KinectIO::m_instance;

}

KinectIO::KinectIO()
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

	// Init freenect stuff
	m_freenect = new Freenect::Freenect;

	m_grabber = &m_freenect->createDevice<KinectGrabber>(0);
	m_grabber->setDepthFormat(FREENECT_DEPTH_11BIT);
	m_grabber->startVideo();
	m_grabber->startDepth();
}

KinectIO::~KinectIO()
{
	//delete m_freenect;
}

PointBufferPtr KinectIO::getBuffer()
{
	// Get depth image from sensor
	std::vector<short> depthImage(480 * 680, 0);
	m_grabber->getDepthImage(depthImage);

	std::vector<uint8_t> colorImage(480 * 680 * 3, 0);
	m_grabber->getColorImage(colorImage);

	std::set<int> nans;
	for(size_t i = 0; i < depthImage.size(); i++)
	{
		if(isnan(depthImage[i])) nans.insert(i);
	}

	// Return null pointer if no image was grabbed
	if(depthImage.size() == 0) return PointBufferPtr();

	size_t numPoints = depthImage.size() - nans.size();

	// Convert depth image into point cloud
	PointBufferPtr buffer(new PointBuffer);
	floatArr points(new float[numPoints * 3]);
	ucharArr colors(new unsigned char[numPoints * 3]);

	int i,j;
	int index = 0;
	int c = 0;
	for (i = 0; i < 480; i++) {
		for (j = 0; j < 640; j++) {

			if(nans.find(c) == nans.end())
			{
				Eigen::Vector4f v;
				v << j, i, (float)(depthImage[i * 640 + j]), 1.0f;
				v = m_depthMatrix.transpose() * v;

				points[3 * index    ] = v(0) / v(3);
				points[3 * index + 1] = v(1) / v(3);
				points[3 * index + 2] = v(2) / v(3);

				colors[3 * index    ] = colorImage[3 * c    ];
				colors[3 * index + 1] = colorImage[3 * c + 1];
				colors[3 * index + 2] = colorImage[3 * c + 2];
				index++;
			}
			c++;
		}
	}

	buffer->setPointArray(points, numPoints);
	buffer->setColorArray(colors, numPoints);
	return buffer;
}

} // namespace lvr2
