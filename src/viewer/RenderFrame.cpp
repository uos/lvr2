/*
 * RenderFrame.cpp
 *
 *  Created on: 27.08.2008
 *      Author: twiemann
 */

#include "RenderFrame.h"
#include "RenderFrameMoc.cpp"

RenderFrame::RenderFrame(QWidget* parent, EventHandler* e) : QGLWidget(parent) {
	eventHandler = e;
	setFocusPolicy(Qt::StrongFocus);
}

RenderFrame::~RenderFrame() {

}

void RenderFrame::createBGList()
{
	m_bgListIndex = glGenLists(1);
	glNewList(m_bgListIndex, GL_COMPILE);
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glDisable(GL_LIGHTING);

	glBegin(GL_QUADS);

	//light blue
	glColor3f(1.0,1.0,1.0);
	glVertex2f(-1.0,0.0);
	glVertex2f(1.0, 0.0);

	//blue color
	glColor3f(0.2,0.2,0.2);
	glVertex2f(1.0, -1.0);
	glVertex2f(-1.0, -1.0);
	glEnd();

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
	glEndList();
}

void RenderFrame::initializeGL(){

	glClearColor(1.0, 1.0, 1.0, 1.0);
	glEnable(GL_DEPTH_TEST);

	glMatrixMode(GL_MODELVIEW);
	glPolygonMode (GL_FRONT_AND_BACK, GL_FILL);

	glEnable(GL_LIGHTING);
	float on[1] = {1.0f};
	//glLightModelfv(GL_LIGHT_MODEL_TWO_SIDE, on);

	GLfloat lightOnePosition[] = {1.0, 1.0, 1.0, 1.0};
	GLfloat lightOneColor[] = {0.10, 0.10, 0.10, 1.0};

	glShadeModel (GL_SMOOTH);
	glHint (GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

	glLightfv (GL_LIGHT0, GL_POSITION, lightOnePosition);
	glLightfv (GL_LIGHT0, GL_AMBIENT, lightOneColor);
	glEnable (GL_LIGHT0);



	float r = 0.0;
	float g = 1.0;
	float b = 0.0;

	float mat_specular[4];
	float mat_ambient[4];
	float mat_diffuse[4];

	float mat_shininess = 50;

	mat_specular[0] = 0.7f; mat_ambient[0]  = 0.5f * r; mat_diffuse[0]  = r;
	mat_specular[1] = 0.7f; mat_ambient[1]  = 0.5f * g; mat_diffuse[1]  = g;
	mat_specular[2] = 0.7f; mat_ambient[2]  = 0.5f * b; mat_diffuse[2]  = b;
	mat_specular[3] = 1.0f; mat_ambient[3]  = 1.0f; mat_diffuse[3]  = 1.0f;

	glEnable(GL_COLOR_MATERIAL);

	glMaterialfv(GL_FRONT, GL_SPECULAR,  mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, &mat_shininess);
	glMaterialfv(GL_FRONT, GL_AMBIENT,   mat_ambient);
	glMaterialfv(GL_FRONT, GL_DIFFUSE,   mat_diffuse);

	createBGList();

	//eventHandler->createDefaultObjects();

}

void RenderFrame::paintGL(){
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Render background
	glDisable(GL_DEPTH_TEST);
	glCallList(m_bgListIndex);
	glEnable(GL_DEPTH_TEST);

	// Render objects
	eventHandler->render();
}

void RenderFrame::resizeGL(){

}

void RenderFrame::mousePressEvent(QMouseEvent* e){
	pressed_button = e->button();
}


void RenderFrame::mouseMoveEvent(QMouseEvent* e){

	int dx = old_x - e->x();
	int dy = old_y - e->y();


	if(pressed_button == Qt::LeftButton){
		if(fabs(dx) > 2*MOUSE_SENSITY){
			dx > 0 ? eventHandler->camTurnLeft() : eventHandler->camTurnRight();
		}
		if(fabs(dy) > MOUSE_SENSITY){
			dy > 0 ? eventHandler->camMoveForward() : eventHandler->camMoveBackward();
		}
	}

	if(pressed_button == Qt::RightButton){
		if(fabs(dy) > MOUSE_SENSITY){
			dy > 0 ? eventHandler->camLookUp() : eventHandler->camLookDown();
		}
	}

	if(pressed_button == Qt::MidButton){
		if(fabs(dy) > MOUSE_SENSITY){
			dy > 0 ? eventHandler->camMoveDown() : eventHandler->camMoveUp();
		}
	}


	old_x = e->x();
	old_y = e->y();

	update();

}

void RenderFrame::keyPressEvent(QKeyEvent* event){

	switch(event->key()){
	case Qt::Key_Up:
		eventHandler->keyUp();
		break;

	case Qt::Key_Plus:
		eventHandler->camIncTransSpeed();
		break;

	case Qt::Key_Minus:
		eventHandler->camDecTransSpeed();
		break;

	case Qt::Key_F1:
		eventHandler->printCurrentTransformation();
		break;

	case Qt::Key_S:
		eventHandler->keyScale();
		break;

	default:
		event->ignore();
	}
	update();
}

void RenderFrame::mouseReleaseEvent(QMouseEvent* e){

}



