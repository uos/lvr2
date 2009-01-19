/*
 * BoundingBox.cpp
 *
 *  Created on: 22.10.2008
 *      Author: twiemann
 */

#include "BoundingBox.h"

BoundingBox::BoundingBox() {

	n = 0;

	float max_val = 1e15;
	float min_val = -max_val;

	v_min = Vertex(max_val, max_val, max_val);
	v_max = Vertex(min_val, min_val, min_val);
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
	// TODO Auto-generated destructor stub
}
