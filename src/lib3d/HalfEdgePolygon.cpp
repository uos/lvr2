 /*
 * HalfEdgePolygon.cpp
 *
 *  Created on: 19.11.2008
 *      Author: twiemann
 */

#include "HalfEdgePolygon.h"

PolygonVertex::PolygonVertex(){
	index = -1;
	next = -1;
}

PolygonVertex::PolygonVertex(int i, int n){
	index = i;
	next = n;
}

PolygonVertex::PolygonVertex(const PolygonVertex &o){
	index = o.index;
	next = o.next;
}


void HalfEdgePolygon::add_vertex(HalfEdgeVertex* v){

	indices.push_back(v->index);
	vertices.push_back(v);

}

void HalfEdgePolygon::add_face(HalfEdgeFace* face){

	faces.insert(make_pair(face->face_index, face));

}

void HalfEdgePolygon::generate_list(){

	map<unsigned int, HalfEdgeFace* >::iterator it;

	HalfEdgeFace* current_face;
	HalfEdgeFace* current_neighbor;

	HalfEdge*     current_edge;
	HalfEdge*     first_edge;

	for(it = faces.begin(); it != faces.end(); it++){

		current_face = it->second;
		current_edge = current_face->edge;
		first_edge = current_edge;

		do{

			if(current_edge->pair != 0){
				if(current_edge->pair->face != 0){
					current_neighbor = current_edge->pair->face;
					if(faces.find(current_neighbor->face_index) == faces.end()){
						edge_list.push_back(current_edge);
					}
				} else {
					edge_list.push_back(current_edge);
				}
			}

			current_edge = current_edge->next;
		} while(current_edge != first_edge);

	}

}

HalfEdgePolygon::~HalfEdgePolygon() {
	// TODO Auto-generated destructor stub
}
