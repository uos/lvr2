/*
 * KinectIO.cpp
 *
 *  Created on: 20.03.2012
 *      Author: Thomas Wiemann
 */

#include "io/KinectIO.hpp"
#include "io/PointBuffer.hpp"
#include "io/DataStruct.hpp"

#include <Eigen/Dense>

#include <vector>
#include <set>

namespace lvr
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

	m_grabber = &m_freenect->createDevice<lvr::KinectGrabber>(0);
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
	ucharArr colors(new uchar[numPoints * 3]);

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
	buffer->setPointColorArray(colors, numPoints);
	return buffer;
}

} // namespace lvr
