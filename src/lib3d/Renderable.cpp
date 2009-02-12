/*
 * Renderable.cpp
 *
 *  Created on: 26.08.2008
 *      Author: Thomas Wiemann
 */

#include "Renderable.h"

Renderable::Renderable() {
	name = "<NO NAME>";
	listIndex = -1;
	axesListIndex = -1;
	visible = true;
	//rotation_speed = 0.0;
	rotation_speed = 0.02f;
	//translation_speed = 0.0;
	translation_speed = 10.0f;

	show_axes = false;
	active = false;

	x_axis = Vertex(1.0f, 0.0f, 0.0f);
	y_axis = Vertex(0.0f, 1.0f, 0.0f);
	z_axis = Vertex(0.0f, 0.0f, 1.0f);

	scale_factor = 1.0f;

	computeMatrix();
	compileAxesList();

}

Renderable::Renderable(Matrix4 m, string n){
	transformation  = m;
	name = n;
	listIndex = -1;
	axesListIndex = -1;
	visible = true;
	//rotation_speed = 0.0;
	rotation_speed = 0.02f;
	//translation_speed = 0.0;
	translation_speed = 10.0f;

	show_axes = false;
	active = false;

	scale_factor = 1.0f;

	setTransformationMatrix(m);
	computeMatrix();
	compileAxesList();
}

Renderable::Renderable(const Renderable& other){
	transformation = other.transformation;
	name = other.name;
	listIndex = other.listIndex;
	axesListIndex = other.axesListIndex;
	visible = other.visible;
	rotation_speed = other.translation_speed;
	translation_speed = other.rotation_speed;

	show_axes = other.show_axes;
	active = other.active;

	x_axis = other.x_axis;
	y_axis = other.y_axis;
	z_axis = other.z_axis;

	scale_factor = other.scale_factor;

	computeMatrix();
	compileAxesList();
}

Renderable::Renderable(string n){
	name = n;
	visible = true;
	listIndex = -1;
	axesListIndex = -1;
	visible = true;
	//rotation_speed = 0.0;
	rotation_speed = 0.02f;
	//translation_speed = 0.0;
	translation_speed = 10.0f;

	show_axes = false;
	active = false;

	x_axis = Vertex(1.0f, 0.0f, 0.0f);
	y_axis = Vertex(0.0f, 1.0f, 0.0f);
	z_axis = Vertex(0.0f, 0.0f, 1.0f);

	scale_factor = 1.0f;

	computeMatrix();
	compileAxesList();

}

void Renderable::scale(float d){

	scale_factor = d;

	float m0  = transformation[0 ] * d;
	float m5  = transformation[5 ] * d;
	float m10 = transformation[10] * d;

	transformation.set(0 , m0 );
	transformation.set(5 , m5 );
	transformation.set(10, m10);
}

void Renderable::yaw(bool invert){
	float speed;
	invert ? speed = rotation_speed : speed = -rotation_speed;

	Quaternion q;
	q.fromAxis(x_axis, speed);
	y_axis = q * y_axis;
	z_axis = q * z_axis;
	computeMatrix();
}

void Renderable::pitch(bool invert){
	float speed;
	invert ? speed = rotation_speed : speed = -rotation_speed;

	Quaternion q;
	q.fromAxis(y_axis, speed);
	x_axis = q * x_axis;
	z_axis = q * z_axis;

	computeMatrix();
}

void Renderable::roll(bool invert){
	float speed;
	invert ? speed = rotation_speed : speed = -rotation_speed;

	Quaternion q;
	q.fromAxis(z_axis, speed);
	x_axis = q * x_axis;
	y_axis = q * y_axis;

	computeMatrix();
}

void Renderable::rotX(bool invert){

	float speed;
	invert ? speed = rotation_speed : speed = -rotation_speed;

	Quaternion q;
	q.fromAxis(Vertex(1.0, 0.0, 0.0), speed);

	x_axis = q * x_axis;
	y_axis = q * y_axis;
	z_axis = q * z_axis;

	computeMatrix();

}

void Renderable::rotY(bool invert){
	float speed;
	invert ? speed = rotation_speed : speed = -rotation_speed;

	Quaternion q;
	q.fromAxis(Vertex(0.0, 1.0, 0.0), speed);

	x_axis = q * x_axis;
	y_axis = q * y_axis;
	z_axis = q * z_axis;

	computeMatrix();

}

void Renderable::rotZ(bool invert){
	float speed;
	invert ? speed = rotation_speed : speed = -rotation_speed;

	Quaternion q;
	q.fromAxis(Vertex(0.0, 0.0, 1.0), speed);

	x_axis = q * x_axis;
	y_axis = q * y_axis;
	z_axis = q * z_axis;

	computeMatrix();

}

void Renderable::accel(bool invert){

	Normal direction(z_axis);
	if(invert) direction = direction * -1;
	position = position + direction * translation_speed;
	cout << position;

	computeMatrix();
}

void Renderable::lift(bool invert){
	Normal direction(y_axis);
	if(invert) direction = direction * -1;
	position = position + direction * translation_speed;
	cout << position;

	computeMatrix();
}

void Renderable::strafe(bool invert){

	Normal direction(x_axis);
	if(invert) direction = direction * -1;
	position = position + direction * translation_speed;
	cout << position;

	computeMatrix();
}

Renderable::~Renderable() {

}

void Renderable::computeMatrix(){

	Matrix4 m;

	m.set(1 , x_axis.y);
	m.set(2 , x_axis.z);

	m.set(4 , y_axis.x);
	m.set(6 , y_axis.z);

	m.set(8 , z_axis.x);
	m.set(9 , z_axis.y);

	m.set(12, position.x);
	m.set(13, position.y);
	m.set(14, position.z);

	if(scale_factor == 1.0){
		m.set(0 , x_axis.x);
		m.set(5 , y_axis.y);
		m.set(10, z_axis.z);
	} else {
		m.set(0 , scale_factor * x_axis.x);
		m.set(5 , scale_factor * y_axis.y);
		m.set(10, scale_factor * z_axis.z);
	}

	transformation = m;
}

void Renderable::compileAxesList(){

	axesListIndex = glGenLists(1);
	glNewList(axesListIndex, GL_COMPILE);

	glDisable(GL_LIGHTING);
	glBegin(GL_LINES);
	glColor3f(1.0, 0.0, 0.0);
	glVertex3f(-1000000, 0.0, 0.0);
	//glVertex3f(0.0, 0.0, 0.0);
	glVertex3f(+1000000, 0.0, 0.0);

	glColor3f(0.0, 1.0, 0.0);
	glVertex3f(0.0, -1000000, 0.0);
	//glVertex3f(0.0, 0.0, 0.0);
	glVertex3f(0.0, +1000000, 0.0);

	glColor3f(0.0, 0.0, 1.0);
	glVertex3f(0.0, 0.0, -100000);
	//glVertex3f(0.0, 0.0, 0.0);
	glVertex3f(0.0, 0.0, +100000);

	glEnd();
	glEnable(GL_LIGHTING);
	glEnd();

	glEndList();

}

void Renderable::setTransformationMatrix(Matrix4 m){

	transformation = m;
	x_axis = Normal(m[0 ], m[1 ], m[2 ]);
	y_axis = Normal(m[4 ], m[5 ], m[6 ]);
	z_axis = Normal(m[8 ], m[9 ], m[10]);

	position.x = m[12];
	position.y = m[13];
	position.z = m[14];

}

void Renderable::transform(){

}
