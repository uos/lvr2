/*
 * Viewport.cpp
 *
 *  Created on: 26.08.2008
 *      Author: twiemann
 */

#include "Viewport.h"

Viewport::Viewport(int w, int h) {

	screen_width = w;
	screen_height = h;

	cam_position_perspective = Vertex(0.0, 0.0, -100.0);
	cam_position_top = Vertex(0.0, 1000.0, 0.0);
	angles = Vertex(0.0, 0.0, 0.0);
	look_at = Vertex(0.0, 0.0, 1.0);
	view_up = Vertex(0.0, 1.0, 0.0);

	trans_speed = 1.0;
	rot_speed = 0.01;

	projection_mode = PERSPECTIVE;
}

void Viewport::camMoveForward(){

  if(projection_mode == PERSPECTIVE){
    cam_position_perspective.x += trans_speed * sin(angles.y);
    cam_position_perspective.z += trans_speed * cos(angles.y);
  }

  if(projection_mode == TOPVIEW){
	 cam_position_top.y += trans_speed;
  }

}

void Viewport::camMoveBackward(){

  if (projection_mode == PERSPECTIVE){
    cam_position_perspective.x -= trans_speed * sin(angles.y);
    cam_position_perspective.z -= trans_speed * cos(angles.y);
  }

  if(projection_mode == TOPVIEW){
    cam_position_top.y -= trans_speed;
  }
}

void Viewport::camMoveUp(){
  cam_position_perspective.y += trans_speed;
}

void Viewport::camMoveDown(){
  cam_position_perspective.y -= trans_speed;
}

void Viewport::camTurnLeft(){

  if(projection_mode == PERSPECTIVE) angles.y += rot_speed;

  if(projection_mode == TOPVIEW) cam_position_top.x -= trans_speed;

}

void Viewport::camTurnRight(){
  if (projection_mode == PERSPECTIVE) angles.y -= rot_speed;
  if (projection_mode == TOPVIEW)  cam_position_top.x += trans_speed;
}

void Viewport::camLookUp(){
  if (projection_mode == PERSPECTIVE && angles.x < 1.5707) angles.x += rot_speed;
  if (projection_mode == TOPVIEW) cam_position_top.z += trans_speed;

}

void Viewport::camLookDown(){
  if (projection_mode == PERSPECTIVE && angles.x > -1.5707) angles.x -= rot_speed;
  if (projection_mode == TOPVIEW) cam_position_top.z -= trans_speed;

}

void Viewport::resize(int w, int h){

  float ratio = 1.0* w / h;

  screen_width = w;
  screen_height = h;

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glViewport(0, 0, w, h);
  if(projection_mode == PERSPECTIVE) gluPerspective(45,ratio,1,100000);
  glMatrixMode(GL_MODELVIEW);

}

void Viewport::topView(){

  projection_mode = TOPVIEW;

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  glOrtho(- 1000,
		+ 1000,
		- 1000,
		+ 1000,
		1.0,
		+ 100000);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
}

void Viewport::perspectiveView(){

  projection_mode = PERSPECTIVE;
  resize(screen_width, screen_height);

}

void Viewport::applyTransformations(){

	switch(projection_mode){
	case PERSPECTIVE:
		look_at.x = cam_position_perspective.x + sin(angles.y);
		look_at.z = -cam_position_perspective.z - cos(angles.y);
		look_at.y = cam_position_perspective.y + sin(angles.x);

		glLoadIdentity();
		gluLookAt(cam_position_perspective.x,
				cam_position_perspective.y,
				cam_position_perspective.z,
				look_at.x, look_at.y, -look_at.z,
				view_up.x, view_up.y, view_up.z);
		break;

	case TOPVIEW:

		glLoadIdentity();
		glRotatef(0.5 * 3.1415926, 0, 1, 0);
		glTranslatef(cam_position_top.x, cam_position_top.y, cam_position_top.z);
		gluLookAt(0.0, 1000, 0.0,
				0, 0, -1,
				0, 1, 0);

		break;
	}

}

Viewport::~Viewport() {

}
