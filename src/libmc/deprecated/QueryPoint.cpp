/*
 * QueryPoint.cpp
 *
 *  Created on: 22.10.2008
 *      Author: Thomas Wiemann
 */

#include "QueryPoint.h"

QueryPoint::QueryPoint() {
	position = Vertex(0.0, 0.0, 0.0);
	distance = 0.0;
}

QueryPoint::QueryPoint(Vertex v){
	position = v;
	distance = 0.0;
}

QueryPoint::QueryPoint(Vertex v, float d){
	position = v;
	distance = d;
}

QueryPoint::QueryPoint(const QueryPoint &o){
	position = o.position;
	distance = o.distance;
}

QueryPoint::~QueryPoint() {
	// TODO Auto-generated destructor stub
}
