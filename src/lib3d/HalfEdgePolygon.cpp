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



PolygonEdge::PolygonEdge(HalfEdge* edge)
{
	v1 = edge->start->position;
	v2 = edge->end->position;

	index1 = edge->start->index;
	index2 = edge->end->index;

	active = true;

}
PolygonEdge::PolygonEdge(const PolygonEdge &o)
{

	v1 = o.v1;
	v2 = o.v2;

	index1 = o.index1;
	index2 = o.index2;

	active = o.active;

}



void HalfEdgePolygon::add_vertex(HalfEdgeVertex* v){

	//indices.push_back(v->index);
	//vertices.push_back(v);

}

PolygonEdge* HalfEdgePolygon::find_edge(HalfEdge* edge)
{
	return 0;

}

void HalfEdgePolygon::add_face(HalfEdgeFace* face, HalfEdge* edge){

	faces.insert(face);

}

HalfEdgeFace* HalfEdgePolygon::find_adj_face(HalfEdge* edge)
{
	if(edge->pair != 0){
		return edge->pair->face;
	} else {
		return 0;
	}
}


void HalfEdgePolygon::fuse_edges(){

	multiset<HalfEdgeFace*>::iterator it;

	HalfEdge* start;
	HalfEdge* current;

	HalfEdgeFace* nb;

	for(it = faces.begin(); it != faces.end(); it++){

		start = (*it)->edge;
		current = start;

		do{
			nb = find_adj_face(current);
			if(nb == 0)
			{
				edges.insert(current);
			} else {
				if(faces.find(nb) == faces.end() ) edges.insert(current);
			}
			current = current->next;
		} while(current != start);

	}


}

void HalfEdgePolygon::generate_list(){


}

void HalfEdgePolygon::test()
{

}

HalfEdgePolygon::~HalfEdgePolygon() {
	// TODO Auto-generated destructor stub
}
