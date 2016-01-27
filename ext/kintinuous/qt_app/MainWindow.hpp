/*
 * MainWindow.hpp
 *
 *  Created on: Jan 26, 2016
 *      Author: twiemann
 */

#ifndef EXT_KINTINUOUS_QT_APP_MAINWINDOW_HPP_
#define EXT_KINTINUOUS_QT_APP_MAINWINDOW_HPP_

#include <QtGui>

#include <kfusion/kinfu.hpp>
#include <io/capture.hpp>

#include "KinfuMainWindow.h"


/*
 *
 */
class MainWindow : public QMainWindow, public Ui::MainWindow
{
	Q_OBJECT
public:
	MainWindow(QMainWindow* parent = 0);
	virtual ~MainWindow();

public Q_SLOTS:
	void pollGPUData();
	void finalizeMesh();

private:
	OpenNISource*			m_openNISource;
	kfusion::KinFu::Ptr		m_kinfu;
	QTimer*		 			m_timer;
	cuda::Image 			m_viewImage;
	cuda::Depth 			m_depth_device;
	cv::Mat 				m_deviceImg;


};

#endif /* EXT_KINTINUOUS_QT_APP_MAINWINDOW_HPP_ */
