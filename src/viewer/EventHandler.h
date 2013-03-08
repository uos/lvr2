/*
 * EventHandler.h
 *
 *  Created on: 27.08.2008
 *      Author: twiemann
 */

#ifndef EVENTHANDLER_H_
#define EVENTHANDLER_H_

#include <iostream>
#include <sstream>
using namespace std;

#include <QtGui>

#include "matrixdialog.h"

#include "Matrix4.h"
#include "GroundPlane.h"
#include "CoordinateAxes.h"
#include "Tube.h"
#include "PointCloud.h"
#include "NormalCloud.h"
#include "TriangleMesh.h"
#include "PolygonMesh.h"

#include "ViewerWindow.h"
#include "ObjectHandler.h"
#include "Viewport.h"

class ViewerWindow;

class EventHandler : public QObject{

	Q_OBJECT

public:
	EventHandler(ViewerWindow*);

	virtual ~EventHandler();

	inline void render();
	void resize_event(int w, int h);

	void camTurnLeft();
	void camTurnRight();
	void camLookUp();
	void camLookDown();
	void camMoveForward();
	void camMoveBackward();
	void camMoveUp();
	void camMoveDown();
	void camIncTransSpeed();
	void camDecTransSpeed();

	void keyScale();
	void keyUp();

	void createDefaultObjects();

	void showStatusMessage();
	void printCurrentTransformation();

	void loadObject(string filename);

public slots:
	void action_file_open();
	void action_topView();
	void action_perspectiveView();
	void action_enterMatrix();
	void action_editObjects(QListWidgetItem* item);
	void action_objectSelected(QListWidgetItem* item);

	void transform_from_file();
	void touchpad_transform(int, double);

signals:
	void updateGLWidget();

private:

	void init();

	ViewerWindow* mainWindow;
	ObjectHandler* objectHandler;
	Viewport* viewport;

	int screen_width;
	int screen_height;

	Matrix4 transform_to_gl;

	Tube* tube;
};


inline void EventHandler::render(){
	//Update status bar
	showStatusMessage();
	//Transform SLAM-coordinates to OpenGL-Coordinates
	glMultMatrixf(transform_to_gl.getData());
	//Render objects
	objectHandler->renderObjects();
	//Create viewport
	viewport->applyTransformations();
}

#endif /* EVENTHANDLER_H_ */
