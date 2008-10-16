/*
 * RenderFrame.h
 *
 *  Created on: 27.08.2008
 *      Author: twiemann
 */

#ifndef RENDERFRAME_H_
#define RENDERFRAME_H_

#include <QGLWidget>

#include "EventHandler.h"

#define MOUSE_SENSITY 2

class EventHandler;

class RenderFrame : public QGLWidget{

	Q_OBJECT

public:

	RenderFrame(QWidget* parent, EventHandler* e);
	virtual ~RenderFrame();

	void initializeGL();
	void paintGL();
	void resizeGL();

	void mousePressEvent(QMouseEvent*);
	void mouseMoveEvent(QMouseEvent*);
	void mouseReleaseEvent(QMouseEvent*);

	void keyPressEvent(QKeyEvent*);


private:

	int old_x;
	int old_y;
	int pressed_button;

	EventHandler* eventHandler;
};

#endif /* RENDERFRAME_H_ */
