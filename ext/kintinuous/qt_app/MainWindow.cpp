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
	KinFuParams params = KinFuParams::default_params();
	m_kinfu = KinFu::Ptr( new KinFu(params) );
	m_openNISource = new OpenNISource;
	m_openNISource->setRegistration(true);
}

MainWindow::~MainWindow()
{
	// TODO Auto-generated destructor stub
}

