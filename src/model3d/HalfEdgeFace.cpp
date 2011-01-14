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

	normal.x = normal.x / 3.0f;
	normal.y = normal.y / 3.0f;
	normal.z = normal.z / 3.0f;

	normal.normalize();
}

void HalfEdgeFace::getVertexNormals(vector<Normal> &n){

	HalfEdgeVertex* start = edge->start;
	HalfEdge* current_edge = edge;
	while(current_edge->end != start){
		n.push_back(current_edge->end->normal);
		normal += current_edge->start->normal;
		current_edge = current_edge->next;
	}

}

void HalfEdgeFace::getVertices(vector<Vertex> &v){

	HalfEdgeVertex* start = edge->start;
	HalfEdge* current_edge = edge;
	do{
		v.push_back(current_edge->end->position);
		current_edge = current_edge->next;
	} while(current_edge->end != start);
	v.push_back(current_edge->end->position);
}

void HalfEdgeFace::getAdjacentFaces(vector<HalfEdgeFace*> &adj){

	HalfEdge* current = edge;
	HalfEdge* pair;
	HalfEdgeFace* neighbor;

	do{
		pair = current->pair;
		if(pair != 0){
			neighbor = pair->face;
			if(neighbor != 0){
				adj.push_back(neighbor);
			}
		}
		current = current->next;
	} while(edge != current);

}

Normal HalfEdgeFace::getFaceNormal(){

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

	return Normal(diff1.cross(diff2));

}

Normal HalfEdgeFace::getInterpolatedNormal(){

	Normal return_normal;

	HalfEdge* start = edge;
	HalfEdge* current_edge = edge;

	int c = 0;
	do{
		return_normal += current_edge->start->normal;
		current_edge = current_edge->next;
		c++;
	} while(current_edge != start);

	return_normal.normalize();
	return return_normal;

}

Vertex HalfEdgeFace::getCentroid(){
	vector<Vertex> vert;
	getVertices(vert);

	Vertex centroid;

	for(size_t i = 0; i < vert.size(); i++){
		centroid += vert[i];
	}

	if(vert.size() > 0){
		centroid.x = centroid.x / vert.size();
		centroid.y = centroid.y / vert.size();
		centroid.z = centroid.z / vert.size();
	} else {
		cout << "Warning: HalfEdgeFace::getCentroid: No vertices found." << endl;
		return Vertex();
	}

	return centroid;
}

