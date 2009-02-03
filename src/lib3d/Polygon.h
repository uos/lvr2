/*
 * Polygon.h
 *
 *  Created on: 03.02.2009
 *      Author: twiemann
 */

#ifndef POLYGON_H_
#define POLYGON_H_

#include <vector>

#include "BaseVertex.h"

using namespace std;

class Polygon {
public:
	Polygon();
	Polygon(const Polygon& other);
	virtual ~Polygon();

	void addVertex(Vertex v){vertices.push_back(v);};
	vector<Vertex> vertices;
};

#endif /* POLYGON_H_ */
