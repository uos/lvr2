/*
 * HalfEdgeFace.cpp
 *
 *  Created on: 03.12.2008
 *      Author: twiemann
 */

#include "HalfEdgeFace.h"

HalfEdgeFace::HalfEdgeFace(const HalfEdgeFace &o){
	edge = o.edge;
	used = o.used;

	for(size_t i = 0; i < o.indices.size(); i++) indices.push_back(o.indices[i]);
	for(int i = 0; i < 3; i++) index[i] = o.index[i];
}

HalfEdgeFace::HalfEdgeFace(){
	edge = 0;
	index[0] = index[1] = index[2] = 0;
}

void HalfEdgeFace::calc_normal(){

	Vertex vertices[3];
	HalfEdgeVertex* start = edge->start;
	HalfEdge* current_edge = edge;

	int c = 0;
	while(current_edge->end != start){
		vertices[c] = current_edge->start->position;
		current_edge = current_edge->next;
		c++;
	}
	Vertex diff1 = vertices[0] - vertices[1];
	Vertex diff2 = vertices[0] - vertices[2];
	normal = Normal(diff1.cross(diff2));
}

void HalfEdgeFace::interpolate_normal(){

	//reset current normal
	normal = Normal();

	HalfEdgeVertex* start = edge->start;
	HalfEdge* current_edge = edge;

	int c = 0;
	while(current_edge->end != start){
		normal += current_edge->start->normal;
		current_edge = current_edge->next;
		c++;
	}

	normal.x = normal.x / 3.0;
	normal.y = normal.y / 3.0;
	normal.z = normal.z / 3.0;

	normal.normalize();


}

