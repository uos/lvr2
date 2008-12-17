/*
 * LinkedTriangle.cpp
 *
 *  Created on: 15.12.2008
 *      Author: twiemann
 */

#include "LinkedTriangle.h"

void LinkedTriangle::changeVertex(int from, int to){

	assert(from == v0 || from == v1 || from == v2);
	if(from == v0){
		v0 = to;
	}
	else if(from == v1){
		v1 = to;
	}
	else if(from == v2){
		v2 = to;
	}

}

LinkedTriangle::LinkedTriangle(const LinkedTriangle &o){
	v0         = o.v0;
	v1         = o.v1;
	v2         = o.v2;
	face_index = o.face_index;
	mesh       = o.mesh;
	active     = o.active;
	normal     = o.normal;
	d          = o.d;
}

