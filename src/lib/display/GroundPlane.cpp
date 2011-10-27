/* Copyright (C) 2011 Uni Osnabrück
 * This file is part of the LAS VEGAS Reconstruction Toolkit,
 *
 * LAS VEGAS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * LAS VEGAS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA
 */


 /*
 * GroundPlane.cpp
 *
 *  Created on: 27.08.2008
 *      Author: Thomas Wiemann
 */

#include "GroundPlane.hpp"

namespace lssr
{

GroundPlane::GroundPlane() : Renderable("Ground Plane"){
	m_listIndex = -1;
	drawGrid(10, 100);
}

GroundPlane::GroundPlane(int increment, int count) : Renderable("Ground Plane"){
	m_listIndex = -1;
	drawGrid(increment, count);
}

GroundPlane::~GroundPlane(){

}

void GroundPlane::drawGrid(int increment, int count){

	m_listIndex = glGenLists(1);

	glNewList(m_listIndex, GL_COMPILE);
	glPushMatrix();
	glPushAttrib ( GL_CURRENT_BIT | GL_ENABLE_BIT | GL_POLYGON_BIT | GL_LINE_BIT ); // save colors and polygon offsets state to later restore.
	glRotatef(-90, 1.0, 0.0, 0.0);
	glBegin ( GL_LINES );
	glEnable ( GL_POLYGON_OFFSET_LINE );
	glEnable ( GL_LINE_SMOOTH );
	glPolygonOffset ( .5,.5 );
	glColor3f ( 0.8f, 0.8f, 1.0f );
	glLineWidth ( 1 );
	for ( int temp = -count; temp <= count; temp +=1 )
	{
		glVertex3i ( temp * increment, -increment * count, 0 );
		glVertex3i ( temp * increment,  increment * count, 0 );
		glVertex3i ( -increment * count, temp * increment, 0 );
		glVertex3i (  increment * count, temp * increment, 0 );
	}
	glLineWidth ( 1.5 );
	glColor3f ( 0.8f, 0.8f, 0.8f);
	for ( int temp = -count; temp <= count; temp +=5 )
	{
		glVertex3i ( temp * increment, -increment * count, 0 );
		glVertex3i ( temp * increment,  increment * count, 0 );
		glVertex3i ( -increment*count, temp * increment, 0 );
		glVertex3i (  increment*count, temp * increment, 0 );
	}
	glDisable ( GL_POLYGON_OFFSET_LINE );
	glEnd();
	glLineWidth ( 1 );
	glDisable ( GL_LINE_SMOOTH );
	glPopMatrix();
	glPopAttrib();
	glEndList();

}

void GroundPlane::render(){
	if(m_visible){
		glDisable(GL_LIGHTING);
		glCallList(m_listIndex);
		glEnable(GL_LIGHTING);
	}

}

void GroundPlane::transform(Matrix4<float> m){
	//To Do: Write transformation code
}

} // namespace lssr

