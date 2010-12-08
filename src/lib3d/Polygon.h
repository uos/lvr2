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


namespace Lib3D{

	class Polygon {
	public:
		Polygon();
		Polygon(const Polygon& other);
		virtual ~Polygon();

		void addVertex(double x,double y,double z){vertices.push_back(Vertex(x,y,z));};
		void addVertex(Vertex v){vertices.push_back(v);};
		vector<Vertex> vertices;
		
		double color_r;
		double color_g;
		double color_b;
		double color_alpha;
	};
}

#endif /* POLYGON_H_ */
