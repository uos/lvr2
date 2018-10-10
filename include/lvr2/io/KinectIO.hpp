/*
 * KinectIO.h
 *
 *  Created on: 20.03.2012
 *      Author: twiemann
 */

#ifndef KINECTIO_H_
#define KINECTIO_H_

#include <lvr2/io/KinectGrabber.hpp>
#include <Eigen/Dense>

namespace lvr2
{

class KinectIO
{
protected:
	KinectIO();

public:
	static KinectIO* instance();

	virtual ~KinectIO();

	PointBufferPtr getBuffer();

private:
	KinectGrabber* 			m_grabber;
	Freenect::Freenect*		m_freenect;
	Eigen::Matrix4f			m_depthMatrix;

	static KinectIO*		m_instance;
};

} // namespace lvr2

#endif /* KINECTIO_H_ */
