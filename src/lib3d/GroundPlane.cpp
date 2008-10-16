/*
 * GroundPlane.cpp
 *
 *  Created on: 27.08.2008
 *      Author: Thomas Wiemann
 */

#include "GroundPlane.h"

GroundPlane::GroundPlane() : Renderable("Ground Plane"){
	listIndex = -1;
	drawGrid(10, 100);
}

GroundPlane::GroundPlane(int increment, int count) : Renderable("Ground Plane"){
	listIndex = -1;
	drawGrid(increment, count);
}

GroundPlane::~GroundPlane(){

}

void GroundPlane::drawGrid(int increment, int count){

	listIndex = glGenLists(1);

	glNewList(listIndex, GL_COMPILE);
	glPushMatrix();
	glPushAttrib ( GL_CURRENT_BIT | GL_ENABLE_BIT | GL_POLYGON_BIT | GL_LINE_BIT ); // save colors and polygon offsets state to later restore.
	glRotatef(-90, 1.0, 0.0, 0.0);
	glBegin ( GL_LINES );
	glEnable ( GL_POLYGON_OFFSET_LINE );
	glEnable ( GL_LINE_SMOOTH );
	glPolygonOffset ( .5,.5 );
	glColor3f ( 0.8, 0.8, 1.0 );
	glLineWidth ( 1 );
	for ( int temp = -count; temp <= count; temp +=1 )
	{
		glVertex3f ( temp*increment, -increment*count, 0 );
		glVertex3f ( temp*increment, increment*count, 0 );
		glVertex3f ( -increment*count, temp*increment, 0 );
		glVertex3f ( increment*count, temp*increment, 0 );
	}
	glLineWidth ( 1.5 );
	glColor3f ( 0.8,0.8,0.8 );
	for ( int temp = -count; temp <= count; temp +=5 )
	{
		glVertex3f ( temp*increment, -increment*count, 0 );
		glVertex3f ( temp*increment, increment*count, 0 );
		glVertex3f ( -increment*count, temp*increment, 0 );
		glVertex3f ( increment*count, temp*increment, 0 );
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
	if(visible){
		glDisable(GL_LIGHTING);
		glCallList(listIndex);
		glEnable(GL_LIGHTING);
	}

}

void GroundPlane::transform(Matrix4 m){
	//To Do: Write transformation code
}

