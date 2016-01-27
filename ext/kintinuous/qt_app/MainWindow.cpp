/*
 * MainWindow.cpp
 *
 *  Created on: Jan 26, 2016
 *      Author: twiemann
 */

#include "MainWindow.hpp"



MainWindow::MainWindow(QMainWindow* parent) : QMainWindow(parent)
{
	setupUi(this);

	// Create Kinfu object
	KinFuParams params = KinFuParams::default_params();
	m_kinfu = KinFu::Ptr( new KinFu(params) );

	// Setup OpenNI Device
	m_openNISource = new OpenNISource;
	string device = "0";
	if(device.find_first_not_of("0123456789") == std::string::npos)
	{
		cuda::setDevice (atoi(device.c_str()));
		cuda::printShortCudaDeviceInfo (atoi(device.c_str()));

		if(cuda::checkIfPreFermiGPU(atoi(device.c_str())))
		{
			std::cout << std::endl << "Kinfu does not support pre-Fermi GPU architectures, and is not built for them by default. Exiting..." << std::endl;
		}
		m_openNISource->open(atoi(device.c_str()));
	}
	else
	{
		m_openNISource->open(device);
		m_openNISource->triggerPause();
	}
	m_openNISource->setRegistration(true);

	// Generate timer for GPU polling
	m_timer = new QTimer(this);
	connect(m_timer, SIGNAL(timeout()), this, SLOT(pollGPUData()));
	m_timer->setInterval(0);
	m_timer->start();

}

void MainWindow::pollGPUData()
{
	KinFu& kinfu = *m_kinfu;
	cv::Mat depth, image, image_copy;
	double time_ms = 0;
	int has_image = 0;

	std::vector<Affine3f> posen;
	std::vector<cv::Mat> rvecs;

	Affine3f best_pose;
	cv::Mat best_rvec,best_image;
	float best_dist=0.0;

	if(!(m_kinfu->hasShifted() && m_kinfu->isLastScan()))
	{
		int has_frame = m_openNISource->grab(depth, image);
		cv::flip(depth, depth, 1);
		cv::flip(image, image, 1);

		if (has_frame == 0)
		{
			std::cout << "Can't grab" << std::endl;
			return;
		}

		// Check if oni file ended
		if (has_frame == 2)
		{
			m_kinfu->performLastScan();
		}
		m_depth_device.upload(depth.data, depth.step, depth.rows, depth.cols);
		has_image = kinfu(m_depth_device);
	}

    const int mode = 4;
    //if (iteractive_mode_)
    //kinfu.renderImage(view_device_, viz.getViewerPose(), mode);
    //else
    m_kinfu->renderImage(m_viewImage, mode);

    cv::Mat view_host;
    view_host.create(m_viewImage.rows(), m_viewImage.cols(), CV_8UC4);
    m_viewImage.download(view_host.ptr<void>(), view_host.step);

    m_displayRaycastLabel->setPixmap(QPixmap::fromImage(QImage((unsigned char*) view_host.data, view_host.cols, view_host.rows, QImage::Format_RGB32)));
    //repaint();
}

MainWindow::~MainWindow()
{
	if(m_timer)
	{
		delete m_timer;
	}

	if(m_openNISource)
	{
		delete m_openNISource;
	}
}

