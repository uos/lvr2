/*
 * KinectIO.cpp
 *
 *  Created on: 20.03.2012
 *      Author: Thomas Wiemann
 */

#include "KinectIO.hpp"
#include "io/PointBuffer.hpp"
#include "io/DataStruct.hpp"

#include "../../ext/Eigen/Dense"

#include <vector>

namespace lssr
{


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

	m_grabber = &m_freenect->createDevice<lssr::KinectGrabber>(0);
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

	// Return null pointer if no image was grabbed
	if(depthImage.size() == 0) return PointBufferPtr();

	// Convert depth image into point cloud
	PointBufferPtr buffer(new PointBuffer);
	floatArr points(new float[depthImage.size() * 3]);

	int i,j;
	int c = 0;
	for (i = 0; i < 480; i++) {
		for (j = 0; j < 640; j++) {

			Eigen::Vector4f v;
			v << j, i, (float)(depthImage[i * 640 + j]), 1.0f;
			v = m_depthMatrix.transpose() * v;

			points[3 * c    ] = v(0) / v(3);
			points[3 * c + 1] = v(1) / v(3);
			points[3 * c + 2] = v(2) / v(3);
			c++;

		}
	}

	buffer->setPointArray(points, 640 * 480);
	return buffer;
}

} // namespace lssr
