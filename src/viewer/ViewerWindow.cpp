/*
 * ViewerWindow.cpp
 *
 *  Created on: 27.08.2008
 *      Author: twiemann
 */

#include "ViewerWindow.h"

ViewerWindow::ViewerWindow() {

	//init variables
	first_resize = true;

	//setup unser interface from editor
	setupUi(this);

	//setup dock windows
	setupDocks();

	//Create event management -> need to be exactly here!
	eventHandler = new EventHandler(this);

	//setup OpenGL widget and enable event management
	setupMainWindow();

	//connect signals and slots
	setupConnections();

}

ViewerWindow::ViewerWindow(string filename){

	//init variables
	first_resize = true;

	//setup unser interface from editor
	setupUi(this);

	//setup dock windows
	setupDocks();

	//Create event management -> need to be exactly here!
	eventHandler = new EventHandler(this);

	//setup OpenGL widget and enable event management
	setupMainWindow();

	//connect signals and slots
	setupConnections();

	//Load object
	eventHandler->loadObject(filename);
}

void ViewerWindow::setupMainWindow(){

	//Create render context using the geometry information
	//from the dummy frame in the ui file
	renderFrame = new RenderFrame(this, eventHandler);
	renderFrame->setGeometry(centralwidget->geometry());
	setCentralWidget(renderFrame);

	//Create status bar
	QStatusBar* statusBar = new QStatusBar(this);
	setStatusBar(statusBar);

	setVisible(true);

	//Init default objects and frame geometry
	eventHandler->createDefaultObjects();
	eventHandler->resize_event(renderFrame->width(), renderFrame->height());
}

void ViewerWindow::setupDocks(){

	//Create sub-windows
	object_dialog = new QDockWidget(this);
	object_dialog_ui = new Ui::ObjectDialog;
	object_dialog_ui->setupUi(object_dialog);

	addDockWidget(Qt::LeftDockWidgetArea, object_dialog);
	menuTools->addAction(object_dialog->toggleViewAction());

	QDockWidget* move_dock = new QDockWidget(this);
	Ui::MoveDock* move_dock_ui = new Ui::MoveDock;
	move_dock_ui->setupUi(move_dock);

	//Create touch pad for move dock
	touch_pad = new TouchPad(move_dock, move_dock_ui);
	touch_pad->setGeometry(move_dock_ui->touchFrame->geometry());

	addDockWidget(Qt::LeftDockWidgetArea, move_dock);
	menuTools->addAction(move_dock->toggleViewAction());

}

void ViewerWindow::setupConnections(){

	QObject::connect(eventHandler, SIGNAL(updateGLWidget()),
			         renderFrame, SLOT(update()));

	QObject::connect(action_open, SIGNAL(activated()),
			         eventHandler, SLOT(action_file_open()));

	QObject::connect(action_topView, SIGNAL(activated()),
			         eventHandler, SLOT(action_topView()));

	QObject::connect(action_perspective, SIGNAL(activated()),
			         eventHandler, SLOT(action_perspectiveView()));

	QObject::connect(action_transformObject, SIGNAL(activated()),
			         eventHandler, SLOT(action_enterMatrix()));

	QObject::connect(action_loadMatrix, SIGNAL(activated()),
			         eventHandler, SLOT(transform_from_file()));

	object_dialog->connect(object_dialog_ui->listWidget, SIGNAL(itemChanged(QListWidgetItem *)),
			         eventHandler, SLOT(action_editObjects(QListWidgetItem *)));

	object_dialog->connect(object_dialog_ui->listWidget, SIGNAL(itemClicked(QListWidgetItem *)),
			         eventHandler, SLOT(action_objectSelected(QListWidgetItem *)));

	touch_pad->connect(touch_pad, SIGNAL(transform(int, double)),
			         eventHandler, SLOT(touchpad_transform(int, double)));

}

void ViewerWindow::resizeEvent(QResizeEvent* e){
	eventHandler->resize_event(renderFrame->geometry().width(),
			                   renderFrame->geometry().height());
}

ViewerWindow::~ViewerWindow() {
	delete eventHandler;
}
