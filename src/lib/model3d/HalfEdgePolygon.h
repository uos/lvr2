/*
 * HalfEdgePolygon.h
 *
 *  Created on: 19.11.2008
 *      Author: Thomas Wiemann
 */

#ifndef HALFEDGEPOLYGON_H_
#define HALFEDGEPOLYGON_H_

#ifdef _MSC_VER
#include <hash_map>
#else
#include <ext/hash_map>
using __gnu_cxx::hash_map;
#endif

#include "HalfEdgeMesh.h"

#include <vector>
#include <map>
#include <set>

using std::vector;
using std::map;
using std::set;

class HalfEdgeFace;
class HalfEdge;
class HalfEdgeVertex;

class PolygonEdge{
public:
	PolygonEdge() : index1(0), index2(2), active(true) {};
	PolygonEdge(HalfEdge*);
	PolygonEdge(const PolygonEdge &o);

	Vertex v1;
	Vertex v2;

	int index1;
	int index2;

	bool active;
};

class Contour{
public:
	Contour();
	Contour(HalfEdgeVertex* v);
	Contour(const Contour &other);

	Contour split(HalfEdgeVertex* v);

	bool contains(HalfEdge* e);

	void add(HalfEdge* v) {edges.insert(v);};

	multiset<HalfEdge* > edges;
};

class PolygonVertex{
public:
	PolygonVertex();
	PolygonVertex(int index, int next);
	PolygonVertex(const PolygonVertex &o);

	int index;
	int next;
};

class HalfEdgePolygon {
public:

	HalfEdgePolygon() : number_of_used_edges(0){};
	HalfEdgePolygon(HalfEdgeFace* first);
	virtual ~HalfEdgePolygon();

	void add_face(HalfEdgeFace* face, HalfEdge* edge);
	void generate_list();
	void fuse_edges();

	bool trace(HalfEdge* start, Contour &c);


	void split(HalfEdge* new_start,
			   HalfEdge* first_start,
			   Contour &contour, bool &fuse);

	void fuse_contours(Contour &c1, Contour c2);
	void remove_contour(Contour &c);

	HalfEdge*      find_edge(HalfEdge* edge);
	HalfEdgeFace*  find_adj_face(HalfEdge* edge);

	void add_vertex(HalfEdgeVertex* v);
	void test();

	int gee(set<HalfEdge*> &v, HalfEdgeVertex*); //Number of emitting edges

	int number_of_used_edges;

	multiset<HalfEdgeFace*> 				faces;
	multimap<HalfEdgeVertex*, HalfEdge*>    edges;

	vector<Contour>         contours;

};

typedef multimap<HalfEdgeVertex*, HalfEdge*> EdgeMap;
typedef multimap<HalfEdgeVertex*, HalfEdge*>::iterator EdgeMapIterator;


#endif /* HALFEDGEPOLYGON_H_ */
