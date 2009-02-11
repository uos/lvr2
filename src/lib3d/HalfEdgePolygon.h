/*
 * HalfEdgePolygon.h
 *
 *  Created on: 19.11.2008
 *      Author: Thomas Wiemann
 */

#ifndef HALFEDGEPOLYGON_H_
#define HALFEDGEPOLYGON_H_

#include "HalfEdgeMesh.h"

#include <ext/hash_map>
using __gnu_cxx::hash_map;

#include <vector>
#include <map>
using std::vector;
using std::map;

class HalfEdgeFace;
class HalfEdge;
class HalfEdgeVertex;

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

	void add_face(HalfEdgeFace* face);
	void generate_list();
	void fuse_edges();

	void add_vertex(HalfEdgeVertex* v);


	int texture_index;

	//vector<int> indices;
	//vector<HalfEdgeVertex*> vertices;

	map<unsigned int, HalfEdgeFace* > faces;
	map<HalfEdgeVertex*, HalfEdge*> edge_list;

	vector<vector<Vertex> > contours;

	int number_of_used_edges;



};

#endif /* HALFEDGEPOLYGON_H_ */
