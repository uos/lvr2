/*
 * Renderable.cpp
 *
 *  Created on: 26.08.2008
 *      Author: Thomas Wiemann
 */

#include "Renderable.hpp"

namespace lssr
{

Renderable::Renderable()
{
	m_name = "<NO NAME>";
	m_listIndex = -1;
	m_activeListIndex = -1;
	m_axesListIndex = -1;
	m_visible = true;
	//rotation_speed = 0.0;
	m_rotationSpeed = 0.02f;
	//translation_speed = 0.0;
	m_translationSpeed = 10.0f;

	m_showAxes = false;
	m_active = true;
	m_selected = false;

	m_xAxis = Vertex<float>(1.0f, 0.0f, 0.0f);
	m_yAxis = Vertex<float>(0.0f, 1.0f, 0.0f);
	m_z_Axis = Vertex<float>(0.0f, 0.0f, 1.0f);

	m_scaleFactor = 1.0f;

	m_boundingBox =  new BoundingBox<Vertex<float> >;

	computeMatrix();
	//compileAxesList();

}

Renderable::Renderable(Matrix4<float> m, string n)
{
	m_transformation  = m;
	m_name = n;
	m_listIndex = -1;
	m_axesListIndex = -1;
	m_visible = true;
	//rotation_speed = 0.0;
	m_rotationSpeed = 0.02f;
	//translation_speed = 0.0;
	m_translationSpeed = 10.0f;

	m_showAxes = false;
	m_active = false;

	m_scaleFactor = 1.0f;

	m_boundingBox = 0;

	setTransformationMatrix(m);
	computeMatrix();
	compileAxesList();
}

Renderable::Renderable(const Renderable& other)
{
	m_transformation = other.m_transformation;
	m_name = other.m_name;
	m_listIndex = other.m_listIndex;
	m_axesListIndex = other.m_axesListIndex;
	m_visible = other.m_visible;
	m_rotationSpeed = other.m_translationSpeed;
	m_translationSpeed = other.m_rotationSpeed;

	m_showAxes = other.m_showAxes;
	m_active = other.m_active;

	m_xAxis = other.m_xAxis;
	m_yAxis = other.m_yAxis;
	m_z_Axis = other.m_z_Axis;

	m_scaleFactor = other.m_scaleFactor;

	m_boundingBox = other.m_boundingBox;

	computeMatrix();
	compileAxesList();
}

Renderable::Renderable(string n)
{
	m_name = n;
	m_visible = true;
	m_listIndex = -1;
	m_axesListIndex = -1;
	m_visible = true;
	//rotation_speed = 0.0;
	m_rotationSpeed = 0.02f;
	//translation_speed = 0.0;
	m_translationSpeed = 10.0f;

	m_showAxes = false;
	m_active = false;

	m_xAxis = Vertex<float>(1.0f, 0.0f, 0.0f);
	m_yAxis = Vertex<float>(0.0f, 1.0f, 0.0f);
	m_z_Axis = Vertex<float>(0.0f, 0.0f, 1.0f);

	m_scaleFactor = 1.0f;

	m_boundingBox = 0;

	computeMatrix();
	compileAxesList();

}

void Renderable::scale(float d)
{

	m_scaleFactor = d;

	float m0  = m_transformation[0 ] * d;
	float m5  = m_transformation[5 ] * d;
	float m10 = m_transformation[10] * d;

	m_transformation.set(0 , m0 );
	m_transformation.set(5 , m5 );
	m_transformation.set(10, m10);
}

void Renderable::yaw(bool invert)
{
	float speed;
	invert ? speed = m_rotationSpeed : speed = -m_rotationSpeed;

	Quaternion<float> q;
	q.fromAxis(m_xAxis, speed);
	m_yAxis = q * m_yAxis;
	m_z_Axis = q * m_z_Axis;
	computeMatrix();
}

void Renderable::pitch(bool invert)
{
	float speed;
	invert ? speed = m_rotationSpeed : speed = -m_rotationSpeed;

	Quaternion<float> q;
	q.fromAxis(m_yAxis, speed);
	m_xAxis = q * m_xAxis;
	m_z_Axis = q * m_z_Axis;

	computeMatrix();
}

void Renderable::roll(bool invert)
{
	float speed;
	invert ? speed = m_rotationSpeed : speed = -m_rotationSpeed;

	Quaternion<float> q;
	q.fromAxis(m_z_Axis, speed);
	m_xAxis = q * m_xAxis;
	m_yAxis = q * m_yAxis;

	computeMatrix();
}

void Renderable::rotX(bool invert)
{

	float speed;
	invert ? speed = m_rotationSpeed : speed = -m_rotationSpeed;

	Quaternion<float> q;
	q.fromAxis(Vertex<float>(1.0, 0.0, 0.0), speed);

	m_xAxis = q * m_xAxis;
	m_yAxis = q * m_yAxis;
	m_z_Axis = q * m_z_Axis;

	computeMatrix();

}

void Renderable::rotY(bool invert)
{
	float speed;
	invert ? speed = m_rotationSpeed : speed = -m_rotationSpeed;

	Quaternion<float> q;
	q.fromAxis(Vertex<float>(0.0, 1.0, 0.0), speed);

	m_xAxis = q * m_xAxis;
	m_yAxis = q * m_yAxis;
	m_z_Axis = q * m_z_Axis;

	computeMatrix();

}

void Renderable::rotZ(bool invert)
{
	float speed;
	invert ? speed = m_rotationSpeed : speed = -m_rotationSpeed;

	Quaternion<float> q;
	q.fromAxis(Vertex<float>(0.0, 0.0, 1.0), speed);

	m_xAxis = q * m_xAxis;
	m_yAxis = q * m_yAxis;
	m_z_Axis = q * m_z_Axis;

	computeMatrix();

}

void Renderable::accel(bool invert)
{

	Normal<float> direction(m_z_Axis);
	if(invert) direction = direction * -1;
	m_position = m_position + direction * m_translationSpeed;

	computeMatrix();
}

void Renderable::lift(bool invert)
{
	Normal<float> direction(m_yAxis);
	if(invert) direction = direction * -1;
	m_position = m_position + direction * m_translationSpeed;

	computeMatrix();
}

void Renderable::strafe(bool invert)
{

	Normal<float> direction(m_xAxis);
	if(invert) direction = direction * -1;
	m_position = m_position + direction * m_translationSpeed;
	cout << m_position;

	computeMatrix();
}

Renderable::~Renderable() {

}

void Renderable::computeMatrix(){

	Matrix4<float> m;

	m.set(1 , m_xAxis.y);
	m.set(2 , m_xAxis.z);

	m.set(4 , m_yAxis.x);
	m.set(6 , m_yAxis.z);

	m.set(8 , m_z_Axis.x);
	m.set(9 , m_z_Axis.y);

	m.set(12, m_position.x);
	m.set(13, m_position.y);
	m.set(14, m_position.z);

	if(m_scaleFactor == 1.0)
	{
		m.set(0 , m_xAxis.x);
		m.set(5 , m_yAxis.y);
		m.set(10, m_z_Axis.z);
	}
	else
	{
		m.set(0 , m_scaleFactor * m_xAxis.x);
		m.set(5 , m_scaleFactor * m_yAxis.y);
		m.set(10, m_scaleFactor * m_z_Axis.z);
	}

	m_transformation = m;
}

void Renderable::compileAxesList(){

	m_axesListIndex = glGenLists(1);
	glNewList(m_axesListIndex, GL_COMPILE);

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
	//glEnd();

	glEndList();

}

void Renderable::setTransformationMatrix(Matrix4<float> m)
{

	m_transformation = m;
	m_xAxis = Normal<float>(m[0 ], m[1 ], m[2 ]);
	m_yAxis = Normal<float>(m[4 ], m[5 ], m[6 ]);
	m_z_Axis = Normal<float>(m[8 ], m[9 ], m[10]);

	m_position.x = m[12];
	m_position.y = m[13];
	m_position.z = m[14];

}

void Renderable::transform(){

}


} // namespace lssr
