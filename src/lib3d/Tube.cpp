/*
 * Tube.cpp
 *
 *  Created on: 26.08.2008
 *      Author: Thomas Wiemann
 */

#include "Tube.h"

Tube::Tube() {

}

Tube::Tube(const Tube& other) : Renderable(other){

	radius = other.radius;
	quadric = other.quadric;

}

Tube::Tube(Matrix4 m, float r): Renderable(m, "Tube"){

	radius = r;
	quadric = gluNewQuadric();
}

Tube::Tube(Vertex pos, Vertex a, float r): Renderable("Tube") {

	radius = r;
	quadric = gluNewQuadric();
}


Tube::~Tube() {
	gluDeleteQuadric(quadric);
}

Vertex Tube::calcClosestPoint(Vertex query_point){
	Vertex vector_to_point = query_point - position;
	double scale = vector_to_point * z_axis;
	Vertex projection_on_axis = z_axis * scale;
	Normal projection_to_point = Normal(query_point - projection_on_axis);
	Vertex closest_point = projection_on_axis + Vertex(projection_to_point * radius);
	return closest_point;
}

void Tube::render(){
	if(visible){
		glPushMatrix();
		glMultMatrixd(transformation.getData());
		gluCylinder(quadric, radius, radius, 10000, 100, 1);
		if(active) glCallList(axesListIndex);
		glPopMatrix();
	}
}


