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
	listIndex = glGenLists(1);
	glNewList(listIndex, GL_COMPILE);
	glPushAttrib ( GL_ALL_ATTRIB_BITS ); // save colors and polygon offsets state to later restore.
	const float charWidth = length / 40.0;
	const float charHeight = length / 30.0;
	const float charShift = 1.04 * length;

	glDisable ( GL_LIGHTING );
	glLineWidth ( 2.0 );

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
	drawArrow ( length, 0.01*length );

	color[0] = 1.0f;
	color[1] = 0.7f;
	color[2] = 0.7f;
	color[3] = 1.0f;
	glMaterialfv ( GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color );
	glPushMatrix();
	glRotatef ( 90.0, 0.0, 1.0, 0.0 );
	drawArrow ( length, 0.01*length );
	glPopMatrix();

	color[0] = 0.7f;
	color[1] = 1.0f;
	color[2] = 0.7f;
	color[3] = 1.0f;
	glMaterialfv ( GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color );
	glPushMatrix();
	glRotatef ( -90.0, 1.0, 0.0, 0.0 );
	drawArrow ( length, 0.01*length );
	glPopMatrix();

	glPopAttrib();
	glEndList();
}

void CoordinateAxes::drawArrow ( float length, float radius, int nbSubdivisions )
{
	GLUquadric* quadric = gluNewQuadric();

	if ( radius < 0.0 )
		radius = 0.05 * length;

	const float head = 2.5* ( radius / length ) + 0.1;
	const float coneRadiusCoef = 4.0 - 5.0 * head;

	gluCylinder ( quadric, radius, radius, length * ( 1.0 - head/coneRadiusCoef ), nbSubdivisions, 1 );
	glTranslatef ( 0.0, 0.0, length * ( 1.0 - head ) );
	gluCylinder ( quadric, coneRadiusCoef * radius, 0.0, head * length, nbSubdivisions, 1 );
	glTranslatef ( 0.0, 0.0, -length * ( 1.0 - head ) );
	gluDeleteQuadric ( quadric );
}

void CoordinateAxes::render(){
	if (visible) glCallList(listIndex);
}

void CoordinateAxes::transform(Matrix4 m){

}
