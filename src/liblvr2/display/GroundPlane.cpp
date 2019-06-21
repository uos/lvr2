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
 * GroundPlane.cpp
 *
 *  Created on: 27.08.2008
 *      Author: Thomas Wiemann
 */

#include "lvr2/display/GroundPlane.hpp"

namespace lvr2
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

void GroundPlane::transform(Matrix4<Vec> m){
	//To Do: Write transformation code
}

} // namespace lvr2
