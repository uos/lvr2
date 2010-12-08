/*
 * HalfEdgeVertex.cpp
 *
 *  Created on: 03.12.2008
 *      Author: twiemann
 */

#include "HalfEdgeVertex.h"

HalfEdgeVertex::HalfEdgeVertex(){
	index = -1;
	color = 0;
}

HalfEdgeVertex::HalfEdgeVertex(const HalfEdgeVertex &o){

	index = o.index;
	position = o.position;
	normal = o.normal;

	vector<HalfEdge*>::iterator it;

	in.clear();
	out.clear();

	color = o.color;

	for(size_t i = 0; i < o.in.size(); i++) in.push_back(o.in[i]);
	for(size_t i = 0; i < o.out.size(); i++) out.push_back(o.out[i]);
}
