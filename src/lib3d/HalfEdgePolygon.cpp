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

	//indices.push_back(v->index);
	//vertices.push_back(v);

}

void HalfEdgePolygon::add_face(HalfEdgeFace* face){

	faces.insert(make_pair(face->face_index, face));

}

void HalfEdgePolygon::fuse_edges(){

	vector<Vertex> current_contour;

	HalfEdge* start_edge;
	HalfEdge* current_edge;
	cout << "BEGIN" << endl;

	map<HalfEdgeVertex*, HalfEdge*>::iterator it;

	//Create contours
	do{
		current_contour.clear();
		current_edge = edge_list.begin()->second;
		start_edge = current_edge;

		cout << "START INNER LOOP" << endl;
		do{

			current_contour.push_back(current_edge->start->position);
			it = edge_list.find(current_edge->end);

			if(it == edge_list.end()){
				cout << "NO MATCHING EDGE" << endl;
			}
			else{
				current_edge = it->second;
				edge_list.erase(it);
			}

			number_of_used_edges++;

		} while(current_edge != start_edge);
		cout << "END INNER LOOP" << endl;

		cout << "NEW CONTOUR: " << number_of_used_edges << " / " << edge_list.size() << endl;
		contours.push_back(current_contour);
	} while(number_of_used_edges < edge_list.size());

	cout << "END" << endl;
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
						edge_list.insert(make_pair(current_edge->start, current_edge));
					}
				} else {
					edge_list.insert(make_pair(current_edge->start, current_edge));
				}
			}
			current_edge = current_edge->next;
		} while(current_edge != first_edge);





	}

}

HalfEdgePolygon::~HalfEdgePolygon() {
	// TODO Auto-generated destructor stub
}
