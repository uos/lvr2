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



}

MainWindow::~MainWindow()
{
	// TODO Auto-generated destructor stub
}

