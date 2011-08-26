/*
 * CoordinateAxes.cpp
 *
 *  Created on: 27.08.2008
 *      Author: twiemann
 */

#include "CoordinateAxes.h"

CoordinateAxes::CoordinateAxes() : Renderable("Coordinate System"){
	drawAxes(100);
}

CoordinateAxes::CoordinateAxes(float length): Renderable("Coordinate System"){
	drawAxes(length);
}

CoordinateAxes::~CoordinateAxes() {

}

void CoordinateAxes::drawAxes(float length){
	m_listIndex = glGenLists(1);
	glNewList(m_listIndex, GL_COMPILE);
	glPushAttrib ( GL_ALL_ATTRIB_BITS ); // save colors and polygon offsets state to later restore.
	const float charWidth = length / 40.0f;
	const float charHeight = length / 30.0f;
	const float charShift = 1.04f * length;

	glDisable ( GL_LIGHTING );
	glLineWidth ( 2.0f );

	glBegin ( GL_LINES );
	// The X
	glVertex3f ( charShift,  charWidth, -charHeight );
	glVertex3f ( charShift, -charWidth,  charHeight );
	glVertex3f ( charShift, -charWidth, -charHeight );
	glVertex3f ( charShift,  charWidth,  charHeight );
	// The Y
	glVertex3f ( charWidth, charShift, charHeight );
	glVertex3f ( 0.0,        charShift, 0.0 );
	glVertex3f ( -charWidth, charShift, charHeight );
	glVertex3f ( 0.0,        charShift, 0.0 );
	glVertex3f ( 0.0,        charShift, 0.0 );
	glVertex3f ( 0.0,        charShift, -charHeight );
	// The Z
	glVertex3f ( -charWidth,  charHeight, charShift );
	glVertex3f ( charWidth,  charHeight, charShift );
	glVertex3f ( charWidth,  charHeight, charShift );
	glVertex3f ( -charWidth, -charHeight, charShift );
	glVertex3f ( -charWidth, -charHeight, charShift );
	glVertex3f ( charWidth, -charHeight, charShift );
	glEnd();

	glEnable ( GL_LIGHTING );
	glDisable ( GL_COLOR_MATERIAL );

	float color[4];
	color[0] = 0.7f;
	color[1] = 0.7f;
	color[2] = 1.0f;
	color[3] = 1.0f;
	glMaterialfv ( GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color );
	drawArrow ( length, 0.01f *length );

	color[0] = 1.0f;
	color[1] = 0.7f;
	color[2] = 0.7f;
	color[3] = 1.0f;
	glMaterialfv ( GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color );
	glPushMatrix();
	glRotatef ( 90.0f, 0.0f, 1.0f, 0.0f );
	drawArrow ( length, 0.01f * length );
	glPopMatrix();

	color[0] = 0.7f;
	color[1] = 1.0f;
	color[2] = 0.7f;
	color[3] = 1.0f;
	glMaterialfv ( GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color );
	glPushMatrix();
	glRotatef ( -90.0f, 1.0f, 0.0f, 0.0f );
	drawArrow ( length, 0.01f * length );
	glPopMatrix();

	glPopAttrib();
	glEndList();
}

void CoordinateAxes::drawArrow ( float length, float radius, int nbSubdivisions )
{
	GLUquadric* quadric = gluNewQuadric();

	if ( radius < 0.0f )
		radius = 0.05f * length;

	const float head = 2.5f * ( radius / length ) + 0.1f;
	const float coneRadiusCoef = 4.0f - 5.0f * head;

	gluCylinder ( quadric, radius, radius, length * ( 1.0f - head/coneRadiusCoef ), nbSubdivisions, 1 );
	glTranslatef ( 0.0f , 0.0f, length * ( 1.0f - head ) );
	gluCylinder ( quadric, coneRadiusCoef * radius, 0.0, head * length, nbSubdivisions, 1 );
	glTranslatef ( 0.0f, 0.0f, -length * ( 1.0f - head ) );
	gluDeleteQuadric ( quadric );
}

void CoordinateAxes::render(){
	if (m_visible){
		glPushMatrix();
		glScalef(10.0, 10.0, 10.0);
		glCallList(listIndex);
		glPopMatrix();
	}
}

void CoordinateAxes::transform(Matrix4 m){

}
