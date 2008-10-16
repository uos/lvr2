/*
 * Renderable.h
 *
 *  Created on: 26.08.2008
 *      Author: Thomas Wiemann
 *
 */

#ifndef RENDERABLE_H_
#define RENDERABLE_H_

#ifndef MAC_OSX
#include <GL/gl.h>
#include <GL/glu.h>
#else
#include <OpenGL/GL.h>
#include <OpenGL/GLU.h>
#endif

#include <string>
using namespace std;

#include "Matrix4.h"
#include "Quaternion.h"

class Renderable {
public:
	Renderable();
	Renderable(const Renderable &other);
	Renderable(string name);
	Renderable(Matrix4 m, string name);

	virtual ~Renderable();
	virtual void render() = 0;

	void setTransformationMatrix(Matrix4 m);

	void setName(string s){name = s;};
	void setVisible(bool s){visible = s;};
	void setRotationSpeed(float s){rotation_speed = s;};
	void setTranslationSpeed(float s){translation_speed = s;};
	void setActive(bool a){active = a;};

	void moveX(bool invert = 0)
		{invert ? position.x -= translation_speed: position.x += translation_speed; computeMatrix();};

	void moveY(bool invert = 0)
		{invert ? position.y -= translation_speed: position.y += translation_speed; computeMatrix();};

	void moveZ(bool invert = 0)
		{invert ? position.z -= translation_speed: position.z += translation_speed; computeMatrix();};

	void rotX(bool invert = 0);
	void rotY(bool invert = 0);
	void rotZ(bool invert = 0);

	void yaw(bool invert = 0);
	void pitch(bool invert = 0);
	void roll(bool invert = 0);

	void accel(bool invert = 0);
	void lift(bool invert = 0);
	void strafe(bool invert = 0);


	void showAxes(bool on){ show_axes = on;};

	void compileAxesList();

	string Name() {return name;};
	Matrix4 getTransformation(){return transformation;};


protected:

	virtual void transform();
	void computeMatrix();

	bool visible;
	Matrix4 transformation;
	string name;
	int listIndex;
	int axesListIndex;

	Normal x_axis;
	Normal y_axis;
	Normal z_axis;

	Vertex position;

	float rotation_speed;
	float translation_speed;

	bool show_axes;
	bool active;


};

#endif /* RENDERABLE_H_ */
