/*
 * MainWindow.hpp
 *
 *  Created on: Jan 26, 2016
 *      Author: twiemann
 */

#ifndef EXT_KINTINUOUS_QT_APP_MAINWINDOW_HPP_
#define EXT_KINTINUOUS_QT_APP_MAINWINDOW_HPP_

#include <QtGui>

#include "../kfusion/include/kfusion/kinfu.hpp"
#include <io/capture.hpp>

#include "KinfuMainWindow.h"
#include "MeshUpdateThread.hpp"

#include <vtkSmartPointer.h>
#include <vtkActor.h>
#include <vtkActorCollection.h>
#include <vtkCommand.h>
#include <vtkRenderer.h>
#include <vtkRendererCollection.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkViewport.h>
#include <vtkObjectFactory.h>
#include <vtkGraphicsFactory.h>
#include <vtkOrientationMarkerWidget.h>
#include <vtkAxesActor.h>

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
	void updateMesh(vtkActor*);

private:
	void setupVTK();

	OpenNISource*								m_openNISource;
	kfusion::KinFu::Ptr							m_kinfu;
	QTimer*		 								m_timer;
	cuda::Image 								m_viewImage;
	cuda::Depth 								m_depth_device;
	cv::Mat 									m_deviceImg;

    vtkSmartPointer<vtkRenderer>                m_renderer;
    vtkSmartPointer<vtkRenderWindowInteractor>  m_renderWindowInteractor;
    vtkSmartPointer<vtkOrientationMarkerWidget> m_axesWidget;
    vtkSmartPointer<vtkAxesActor> 				m_axes;
    vtkActor*									m_meshActor;

    MeshUpdateThread*							m_meshThread;
    vector<Affine3f> sample_poses_;
};

#endif /* EXT_KINTINUOUS_QT_APP_MAINWINDOW_HPP_ */
