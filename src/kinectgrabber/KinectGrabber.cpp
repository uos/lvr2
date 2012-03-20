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

KinectGrabber::KinectGrabber(bool autostart)
{
	m_grabber = new pcl::OpenNIGrabber();

	// Create boost function object for grabber callback
	boost::function<void (const pcl::PointCloud<pcl::PointXYZ>::ConstPtr&)> f =
	         boost::bind (&KinectGrabber::update, this, _1);
	cout << "Register" << endl;
	m_grabber->registerCallback(f);
	cout << "Reg Ok" << endl;
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
	cout << timestamp << "New data" << endl;
	m_mutex.lock();
	PointBuffer buffer;
	m_buffer = buffer(cloud);
	m_mutex.unlock();
}

KinectGrabber::~KinectGrabber()
{


}

} // namespace lssr
