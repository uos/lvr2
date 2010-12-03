/*
 * Viewport.h
 *
 *  Created on: 26.08.2008
 *      Author: Thomas Wiemann
 */

#ifndef VIEWPORT_H_
#define VIEWPORT_H_

#include <math.h>

#include "gl.h"
#include "glu.h"

#include <lib3d/BaseVertex.h>


enum {PERSPECTIVE, TOPVIEW};

class Viewport {
public:
	Viewport(int screen_width, int screen_height);
	virtual ~Viewport();

	void setRotationSpeed(float s){rot_speed = s;};
	void setTranslationSpeed(float s){trans_speed = s;};

	void incRotationSpeed(float s){rot_speed += s;};
	void decRotationSpeed(float s){rot_speed -= s;};

	void incTranslationSpeed(float s){trans_speed += s;};
	void decTranslationSpeed(float s){trans_speed -= s;};

	inline Vertex getPosition();
	inline Vertex getOrientation();

	void camMoveForward();
	void camMoveBackward();
	void camMoveUp();
	void camMoveDown();
	void camTurnLeft();
	void camTurnRight();
	void camLookUp();
	void camLookDown();
	void camReset();

	void topView();
	void perspectiveView();

	void resize(int w, int h);

	void applyTransformations();

private:
	float rot_speed;
	float trans_speed;

	int projection_mode;
	int screen_width;
	int screen_height;

	Vertex angles;
	Vertex cam_position_perspective;
	Vertex cam_position_top;
	Vertex view_up;
	Vertex look_at;

};

inline Vertex Viewport::getPosition(){
	if(projection_mode == PERSPECTIVE)
		return cam_position_perspective;
	else
		return cam_position_top;
}

inline Vertex Viewport::getOrientation(){
	return angles;
}

#endif /* VIEWPORT_H_ */
