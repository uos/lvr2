/*
 * KinecGrabber.cpp
 *
 *  Created on: 20.03.2012
 *      Author: Thomas Wiemann
 */

#include "KinectGrabber.hpp"

namespace lssr
{

KinectGrabber::KinectGrabber(bool autostart)
{
	m_grabber = new pcl::OpenNIGrabber();

	// Create boost function object for grabber callback
	boost::function<void (const pcl::PointCloud<pcl::PointXYZ>::ConstPtr&)> f =
	         boost::bind (&KinectGrabber::update, this, _1);

	m_grabber->registerCallback(f);

	// Start thread immediately if autostart is set
	if(autostart)
	{
		start();
	}
}

void KinectGrabber::start()
{
	m_grabber->start();
}

void KinectGrabber::stop()
{
	m_grabber->stop();
}

void KinectGrabber::update(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud)
{

}

KinectGrabber::~KinectGrabber()
{


}

} // namespace lssr
