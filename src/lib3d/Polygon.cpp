/*
 * Polygon.cpp
 *
 *  Created on: 03.02.2009
 *      Author: twiemann
 */

#include "Polygon.h"

using Lib3D::Polygon;

Polygon::Polygon() {
	// TODO Auto-generated constructor stub

}

Polygon::Polygon(const Polygon &other){

	for(size_t i = 0; i < other.vertices.size(); i++){
		vertices.push_back(other.vertices[i]);
	}

}

Polygon::~Polygon() {
	// TODO Auto-generated destructor stub
}
