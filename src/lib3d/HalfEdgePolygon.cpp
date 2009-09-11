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


bool HalfEdgePolygon::trace(HalfEdge* s, Contour &contour)
{

	set<HalfEdge*>::iterator it;
	set<HalfEdge*> out_edges;

	HalfEdge* start   = s;
	HalfEdge* current = s;

	int n;

	cout << "TRACE: " << s << endl;

	do{
		out_edges.clear();

		n = gee(out_edges, current->end);

		if(n == 1){
			contour.add(current);
			current = *out_edges.begin();
		} else if ( n > 1){

			for(it = out_edges.begin(); it != out_edges.end(); it++)
			{
				if( (*it)->end->position == start->start->position)
				{
					current = *it;
				} else {
					Contour c;
					//trace(*it, c);
					return false;
				}
			}

		} else {
			cout << "NULL" << endl;
			return false;
		}

	} while(current->end->position != start->start->position);

	cout << "DELETE: " << endl;

	remove_contour(contour);

	cout << "FINISHED TRACE: " << contour.edges.size() << " " << edges.size() << endl;

	return true;

}

void HalfEdgePolygon::fuse_edges(){

	cout << "FUSE: " << endl;

	EdgeMapIterator it;

	do{
		Contour c;
		it = edges.begin();
		while(!trace(it->second, c) && it != edges.end()){
			it++;
		}
	} while(edges.size() > 0);

}


void HalfEdgePolygon::remove_contour(Contour &c)
{

	multiset<HalfEdge*>::iterator it;
	for(it = c.edges.begin(); it != c.edges.end(); it++)
	{
		edges.erase( (*it)->start);
	}

	//cout << "WARNING: REMOVE CONTOUR NOT IMPLEMENTED!" << endl;

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
				edges.insert(make_pair(current->start, current));
			} else {
				if(faces.find(nb) == faces.end() ) edges.insert(make_pair(current->start, current));
			}
			current = current->next;
		} while(current != start);

	}

}

int HalfEdgePolygon::gee(set<HalfEdge*> &edge_set, HalfEdgeVertex* v)
{
	pair<EdgeMapIterator, EdgeMapIterator> range;
	EdgeMapIterator it;

	cout << edges.count(v) << endl;

	range = edges.equal_range(v);

	if(range.first == edges.end()){
		//cout << "NOT FOUND! " << endl;
		return 0;
	}

	for(it = range.first; it != range.second; ++it)
	{
		edge_set.insert(it->second);
	}

	return (int) edge_set.size();
}

void HalfEdgePolygon::test()
{

}

HalfEdgePolygon::~HalfEdgePolygon() {
	// TODO Auto-generated destructor stub
}
