 /*
 * HalfEdgePolygon.cpp
 *
 *  Created on: 19.11.2008
 *      Author: twiemann
 */

#include "HalfEdgePolygon.h"


Contour::Contour(){

}

Contour::Contour(HalfEdgeVertex* v)
{

}


Contour::Contour(const Contour &other)
{

}

bool Contour::contains(HalfEdge* e)
{
	return edges.find(e) != edges.end();
}



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

HalfEdge* HalfEdgePolygon::find_edge(HalfEdge* edge)
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


bool HalfEdgePolygon::trace(HalfEdge* start, Contour &contour)
{

}

void HalfEdgePolygon::fuse_edges(){
	cout << "FUSE" << endl;
	cout << edges.size() << endl;
	size_t init = edges.size();
	while(edges.size() > 0)
	{
//		HalfEdge* e_start = *edges.begin();
//		cout << "STARTING NEW TRACE FROM POLYGON" << endl;
//		Contour c;
//		trace(e_start, c);
//
//		cout << "Contour size: " << c.edges.size() << endl;

		Contour c;

		multiset<HalfEdge*>::iterator it = edges.begin();
		while(!trace(*it, c)){
			c.edges.clear();
			it++;
			if(it == edges.end()){
				cout << "*** FAIL ***" << endl;
				break;
			}
		}
		cout << "Contour size   : " << c.edges.size() << endl;
		cout << "Number of edges: " << edges.size() << endl;
		cout << "Intial         : " << init << endl;
	}

}


void HalfEdgePolygon::remove_contour(Contour &c)
{
	set<HalfEdge*>::iterator it;
	for(it = c.edges.begin(); it != c.edges.end(); it++)
	{
		edges.erase(*it);
	}
}

void HalfEdgePolygon::fuse_contours(Contour &c1, Contour c2)
{
	multiset<HalfEdge*>::iterator it;
	for(it = c2.edges.begin(); it != c2.edges.end(); it++)
	{
		c1.edges.insert(*it);
	}
}



void HalfEdgePolygon::generate_list(){

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

int HalfEdgePolygon::gee(set<HalfEdge*> &v, HalfEdge* e)
{
	int c = 0;

	vector<HalfEdge*>::iterator e_it;

	for(e_it  = e->end->out.begin();
		e_it != e->end->out.end();
		e_it++)
	{
		HalfEdge* edge = *e_it;
		multiset<HalfEdge*>::iterator f_edge = edges.find(edge);

		if(f_edge != edges.end() && edge != e){
			c++;
			v.insert(*f_edge);
		}
	}

	return c;

}

void HalfEdgePolygon::test()
{
	multiset<HalfEdge*>::iterator it;

	cout << "NEW POLYGON" << endl;
	for(it = edges.begin(); it != edges.end(); it++)
	{
//		HalfEdge* current_edge = *it;
//		vector<HalfEdge*>::iterator e_it;
//		int c = 0;
//		for(e_it  = current_edge->end->out.begin();
//			e_it != current_edge->end->out.end();
//			e_it++)
//		{
//			HalfEdge* e = *e_it;
//			if(edges.find(e) != edges.end()) c++;
//		}
//		cout << c << endl;
		set<HalfEdge*> v;
		cout << gee(v, *it) << endl;
	}
	cout << "END POLYGON" << endl << endl;
}

HalfEdgePolygon::~HalfEdgePolygon() {
	// TODO Auto-generated destructor stub
}
