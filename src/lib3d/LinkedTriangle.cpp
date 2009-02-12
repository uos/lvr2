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
void LinkedTriangle::calculateNormal(){
	assert(mesh);
	Vertex diff1 = mesh->getVertex(v0).position - mesh->getVertex(v1).position;
	Vertex diff2 = mesh->getVertex(v0).position - mesh->getVertex(v2).position;
	normal = Normal(diff1.cross(diff2));

	d = -(normal * mesh->getVertex(v0).position);
}

void LinkedTriangle::interpolateNormal(){
	assert(mesh);

	Normal n1 = mesh->getNormal(v0);
	Normal n2 = mesh->getNormal(v1);
	Normal n3 = mesh->getNormal(v2);

	normal = n1 + n2 + n3;
	normal.normalize();
}

float LinkedTriangle::calculateArea(){
	assert(mesh);

	Vertex diff1 = mesh->getVertex(v0).position - mesh->getVertex(v1).position;
	Vertex diff2 = mesh->getVertex(v2).position - mesh->getVertex(v1).position;
	return 0.5f * diff1.cross(diff2).length();
}

Vertex LinkedTriangle::getVertex(int index){
	assert(mesh);
	assert(index >= 0 && index < 3);

	switch(index){
		case 0: return mesh->getVertex(v0).position; break;
		case 1: return mesh->getVertex(v1).position; break;
		case 2: return mesh->getVertex(v2).position; break;
		default: return Vertex();
	}
}

int LinkedTriangle::getIndex(int index){
	assert(mesh);
	assert(index >= 0 && index < 3);

	switch(index){
	case 0: return v0; break;
	case 1: return v1; break;
	case 2: return v2; break;
	default: return -1;
	}
}






