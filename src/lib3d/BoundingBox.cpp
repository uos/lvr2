/*
 * BoundingBox.cpp
 *
 *  Created on: 22.10.2008
 *      Author: twiemann
 */

#include "BoundingBox.h"

BoundingBox::BoundingBox() {

	n = 0;

	float max_val = 1e15f;
	float min_val = -max_val;

	v_min = Vertex(max_val, max_val, max_val);
	v_max = Vertex(min_val, min_val, min_val);

	pick_objects_list = -1;
	bounding_box_list = -1;
}

BoundingBox::BoundingBox(Vertex v1, Vertex v2){
	n = 0;
	v_min = v1;
	v_max = v2;

}

BoundingBox::BoundingBox(float x_min, float y_min, float z_min,
		                 float x_max, float y_max, float z_max){
	n = 0;
	v_min = Vertex(x_min, y_min, z_min);
	v_max = Vertex(x_max, y_max, z_max);
}


BoundingBox::~BoundingBox() {

}

void BoundingBox::createDisplayLists()
{
	// Delete previously generated display lists

	if(pick_objects_list > -1)
	{
		glDeleteLists(pick_objects_list, 1);
	}

	if(bounding_box_list > -1)
	{
		glDeleteLists(bounding_box_list, 1);
	}

	float radius = 0.5 * min(min(x_size, y_size), z_size);

	// Create new ones
	pick_objects_list = glGenLists(1);
	bounding_box_list = glGenLists(2);

	// Build bounding box list
	glNewList(bounding_box_list, GL_COMPILE);
	glDisable(GL_LIGHTING);
	glColor3f(0.0, 0.0, 0.0);

	glBegin(GL_LINE_LOOP);
	glVertex3f(v_min.x, v_min.y, v_min.z);
	glVertex3f(v_max.x, v_min.y, v_min.z);
	glVertex3f(v_max.x, v_max.y, v_min.z);
	glVertex3f(v_min.x, v_max.y, v_min.z);
	glEnd();

	glBegin(GL_LINE_LOOP);
	glVertex3f(v_min.x, v_min.y, v_max.z);
	glVertex3f(v_max.x, v_min.y, v_max.z);
	glVertex3f(v_max.x, v_max.y, v_max.z);
	glVertex3f(v_min.x, v_max.y, v_max.z);
	glEnd();

	glBegin(GL_LINES);

	glVertex3f(v_min.x, v_min.y, v_min.z);
	glVertex3f(v_min.x, v_min.y, v_max.z);

	glVertex3f(v_max.x, v_min.y, v_min.z);
	glVertex3f(v_max.x, v_min.y, v_max.z);

	glVertex3f(v_min.x, v_max.y, v_min.z);
	glVertex3f(v_min.x, v_max.y, v_max.z);

	glVertex3f(v_max.x, v_max.y, v_min.z);
	glVertex3f(v_max.x, v_max.y, v_max.z);

	glEnd();

	glEnable(GL_LIGHTING);
	glEndList();

	// Create picking objects
//	glNewList(pick_objects_list, GL_COMPILE);
//	glPushMatrix();
//	glTranslatef(centroid.x, centroid.y, centroid.z);
//	glPushMatrix();
//	glColor3f(1.0, 0.0, 0.0);
//	glRotatef(90.0, 0.0, 1.0, 0.0);
//	glutSolidTorus( 0.02 * radius, radius, 5, 30);
//	glPopMatrix();
//	glPushMatrix();
//	glRotatef(90.0, 1.0, 0.0, 0.0);
//	glColor3f(0.0, 1.0, 0.0);
//	glutSolidTorus( 0.02 * radius, radius, 5, 30);
//	glPopMatrix();
//	glColor3f(0.0, 0.0, 1.0);
//	glutSolidTorus( 0.02 * radius, radius, 5, 30);
//	glPopMatrix();
//	glEndList();
}
