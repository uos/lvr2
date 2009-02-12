/*
 * Triangle.cpp
 *
 *  Created on: 11.12.2008
 *      Author: Thomas Wiemann
 */

#include "Triangle.h"

bool Triangle::operator==(const Triangle& t){
	return (v0 == t.v0 &&
			v1 == t.v1 &&
			v2 == t.v2);
}

bool Triangle::operator!=(const Triangle& t){
	return !(*this == t);
}

std::ostream& operator<<(std::ostream& os, const Triangle& t){
	os << "Triangle: " << endl;
	os << "Vertex Indices: " << t.v0 << " " << t.v1 << " " << t.v2 << endl;
	os << t.normal;
	os << "Face Index: " << t.face_index << endl;
	return os;
}

Triangle::Triangle() {
	v0         = -1;
	v1         = -1;
	v2         = -1;
	face_index = -1;
	mesh       = 0;
	active     = true;
	d          = 1e10;
}

Triangle::Triangle(int _v0, int _v1, int _v2){
	v0         = _v0;
	v1         = _v1;
	v2         = _v2;
	face_index = -1;
	mesh       = 0;
	active     = true;
	d          = 1e10;
}

Triangle::Triangle(TriangleMesh* m, int _v0, int _v1, int _v2){
	v0         = _v0;
	v1         = _v1;
	v2         = _v2;
	face_index = -1;
	mesh       = m;
	active     = true;
	d          = 1e10;
	calculateNormal();
}

Triangle::Triangle(const Triangle &o){
	v0         = o.v0;
	v1         = o.v1;
	v2         = o.v2;
	face_index = o.face_index;
	mesh       = o.mesh;
	active     = o.active;
	normal     = o.normal;
	d          = o.d;
}

void Triangle::calculateNormal(){
	assert(mesh);

	Vertex diff1 = mesh->getVertex(v0) - mesh->getVertex(v1);
	Vertex diff2 = mesh->getVertex(v0) - mesh->getVertex(v2);
	normal = Normal(diff1.cross(diff2));
	cout << normal;
}

void Triangle::interpolateNormal(){
	assert(mesh);

	Normal n1 = mesh->getNormal(v0);
	Normal n2 = mesh->getNormal(v1);
	Normal n3 = mesh->getNormal(v2);

	normal = n1 + n2 + n3;
	normal.normalize();
}

float Triangle::calculateArea(){
	assert(mesh);

	Vertex diff1 = mesh->getVertex(v0) - mesh->getVertex(v1);
	Vertex diff2 = mesh->getVertex(v2) - mesh->getVertex(v1);
	return 0.5f * diff1.cross(diff2).length();
}

Vertex Triangle::getVertex(int index){
	assert(mesh);
	assert(index >= 0 && index < 3);

	switch(index){
		case 0: return mesh->getVertex(v0); break;
		case 1: return mesh->getVertex(v1); break;
		case 2: return mesh->getVertex(v2); break;
		default: return Vertex();
	}
}

int Triangle::getIndex(int index){
	assert(mesh);
	assert(index >= 0 && index < 3);

	switch(index){
	case 0: return v0; break;
	case 1: return v1; break;
	case 2: return v2; break;
	default: return -1;
	}
}

Triangle::~Triangle() {
	// TODO Auto-generated destructor stub
}
