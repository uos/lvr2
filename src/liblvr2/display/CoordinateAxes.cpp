/**
 * Copyright (c) 2018, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

 /*
 * CoordinateAxes.cpp
 *
 *  Created on: 27.08.2008
 *      Author: Thomas Wiemann
 */

#include "lvr2/display/CoordinateAxes.hpp"

namespace lvr2
{

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
		glCallList(m_listIndex);
		glPopMatrix();
	}
}

void CoordinateAxes::transform(Matrix4<Vec> m){

}

} // namespace lvr2
