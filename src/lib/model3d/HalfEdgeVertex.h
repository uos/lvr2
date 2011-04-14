/*
 * HalfEdgeVertex.h
 *
 *  Created on: 03.12.2008
 *      Author: twiemann
 */

#ifndef HALFEDGEVERTEX_H_
#define HALFEDGEVERTEX_H_

#include <vector>
using namespace std;

#include "BaseVertex.h"
#include "Normal.h"
#include "HalfEdge.h"

class HalfEdgeVertex{
public:
	HalfEdgeVertex();
	HalfEdgeVertex(const HalfEdgeVertex& o);

	Vertex position;
	Normal normal;

	float color;

	int index;

	vector<HalfEdge*> in;
	vector<HalfEdge*> out;
};

#endif /* HALFEDGEVERTEX_H_ */
