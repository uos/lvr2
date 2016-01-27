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
	m_timer->setInterval(0);

	// Connect signals and slots
	connect(m_pbStart, SIGNAL(pressed()), m_timer, SLOT(start()));
	connect(m_timer, SIGNAL(timeout()), this, SLOT(pollGPUData()));
	connect(m_pbStop, SIGNAL(pressed()), this, SLOT(finalizeMesh()));
}

void  MainWindow::finalizeMesh()
{
	m_kinfu->performLastScan();
}

void MainWindow::pollGPUData()
{
	KinFu& kinfu = *m_kinfu;
	cv::Mat depth, image, image_copy;
	int has_image = 0;

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

    // Raycast image and download from device
    m_kinfu->renderImage(m_viewImage, mode);
    m_deviceImg.create(m_viewImage.rows(), m_viewImage.cols(), CV_8UC4);
    m_viewImage.download(m_deviceImg.ptr<void>(), m_deviceImg.step);

    // Convert cv mat to pixmap and render into label
    m_displayRaycastLabel->setPixmap(
    		QPixmap::fromImage(
    				QImage((unsigned char*) m_deviceImg.data,
    				m_deviceImg.cols,
					m_deviceImg.rows,
					QImage::Format_RGB32)));

    m_displayImageLabel->setPixmap(
    		QPixmap::fromImage(
    				QImage((unsigned char*) image.data,
    				image.cols,
					image.rows,
					QImage::Format_RGB888).rgbSwapped()));

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

	m_kinfu.release();
}

