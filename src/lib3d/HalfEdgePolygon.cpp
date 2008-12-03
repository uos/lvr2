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

HalfEdgePolygon::HalfEdgePolygon(HalfEdgeFace* first_face) {

//	//Add initial vertices to list
//	HalfEdge* first_edge = first_face->edge;
//	HalfEdge* current_edge = first_edge;
//
//	do{
//		PolygonVertex* v = new PolygonVertex(current_edge->start->index, current_edge->end->index);
//		vertices[current_edge->start->index] = v;
//		current_edge = current_edge->next;
//	} while(current_edge != first_edge);
//
//	first = vertices[first_edge->start->index];
//
//
//	//Get pointer to the three triangle vertices
//	HalfEdge* edge = first_face->edge;
//	HalfEdgeVertex* he_vertex1 = edge->start;
//	HalfEdgeVertex* he_vertex2 = edge->end;
//	HalfEdgeVertex* he_vertex3 = edge->next->end;
//
//	//Create polygon vertices
//	PolygonVertex* p_vertex1 = new PolygonVertex(he_vertex1->index, he_vertex2->index);
//	PolygonVertex* p_vertex2 = new PolygonVertex(he_vertex2->index, he_vertex3->index);
//	PolygonVertex* p_vertex3 = new PolygonVertex(he_vertex3->index, he_vertex1->index);
//
//	//Insert polygon vertices into hash table
//	vertices[p_vertex1->index] = p_vertex1;
//	vertices[p_vertex2->index] = p_vertex2;
//	vertices[p_vertex3->index] = p_vertex3;
//
//	first = p_vertex1;

}

void HalfEdgePolygon::add_vertex(HalfEdgeVertex* v){

	indices.push_back(v->index);
	vertices.push_back(v);

}

void HalfEdgePolygon::add_face(HalfEdgeFace* face, HalfEdge* edge){

//	//Test which index face vertex is not already in polygon
//	int new_index = -1;
//
//	hash_map<int, PolygonVertex*>::iterator it;
//
//	HalfEdge* first_edge = edge;
//	HalfEdge* current_edge = first_edge;
//
//	int number_of_new_vertices = 0;
//
//	do{
//		it = vertices.find(current_edge->start->index);
//		if(it == vertices.end()){
//			new_index = current_edge->start->index;
//			number_of_new_vertices++;
//		}
//		current_edge = current_edge->next;
//	} while(current_edge != first_edge);
//
//	if(number_of_new_vertices == 1){
//
//		//Create new polygon vertex and save it
//		PolygonVertex* new_vertex = new PolygonVertex(new_index, first_edge->start->index);
//		vertices[new_index] = new_vertex;
//
//		//Update link of end vertex
//		it = vertices.find(first_edge->end->index);
//		PolygonVertex* end = (*it).second;
//		end->next = new_index;
//
//	} else {
//
//		//Find third vertex of current triangle
//		it = vertices.find(edge->next->end->index);
//		if(it == vertices.end()){
//			cout << "----> Warning: Mesh Error: Third triangle index not in mesh." << endl;
//		}
//		PolygonVertex* v = (*it).second;
//
//		//Find end vertex of the border edge
//		it = vertices.find(edge->end->index);
//		if(it == vertices.end()){
//			cout << "----> Warning: Mesh Error: End index not in mesh." << endl;
//		}
//		PolygonVertex* e = (*it).second;
//
//		//Update link. The start index of the
//		//border edge is no longer part of the
//		//contour polygon.
//		e->next = v->index;
//	}

}

void HalfEdgePolygon::generate_list(vector<int> &list){

//	hash_map<int, PolygonVertex*>::iterator it;
//
//	PolygonVertex* current_vertex = first;
//	int next_index = -1;
//
//	do{
//		it = vertices.find(current_vertex->next);
//		if(it == vertices.end()) cout << "Cannot find index " << current_vertex->index << endl;
//		current_vertex = (*it).second;
//		next_index = current_vertex->next;
//	} while(current_vertex->index != first->index);
//
//	hash_map<int, PolygonVertex*>::iterator it;
//	PolygonVertex* current_vertex = first;
//	list.push_back(first->index);
//	int c = 0;
//	do{
//		it = vertices.find(current_vertex->next);
//		current_vertex = (*it).second;
//		c++;
//		if(c > vertices.size()){
//			cout << "##### HalfEdgePolygon: generate_list(): Warning: Loop Detected. ";
//			cout << "Skipping " << vertices.size() << " vertices" << endl;
//			list.clear();
//			return;
//		}
//		list.push_back(current_vertex->index);
//	} while(current_vertex->next != first->index);

}

HalfEdgePolygon::~HalfEdgePolygon() {
	// TODO Auto-generated destructor stub
}
