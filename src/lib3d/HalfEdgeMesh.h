/*
 * HalfEdgeMesh.h
 *
 *  Created on: 13.11.2008
 *      Author: twiemann
 */

#ifndef HALFEDGEMESH_H_
#define HALFEDGEMESH_H_

#include <vector>
#include <stack>
using namespace std;

#include <ext/hash_map>
using __gnu_cxx::hash_map;

#include "BaseVertex.h"
#include "Normal.h"
#include "StaticMesh.h"

#include "HalfEdgeVertex.h"
#include "HalfEdge.h"
#include "HalfEdgeFace.h"
#include "HalfEdgePolygon.h"



class HalfEdgeVertex;
class HalfEdgeFace;
class HalfEdgePolygon;

class HalfEdgeMesh : public StaticMesh{
public:
	HalfEdgeMesh();
	virtual ~HalfEdgeMesh();

	vector<HalfEdgeFace*>    he_faces;
	vector<HalfEdgeVertex*>  he_vertices;
	vector<HalfEdgePolygon*> hem_polygons;
	int global_index;

	virtual void finalize();
	void write_polygons(string filename);
	void printStats();
	void check_next_neighbor(HalfEdgeFace* f0, HalfEdgeFace* face, HalfEdgePolygon*);
	void extract_borders();


	void create_polygon(vector<int> &polygon, hash_map<unsigned int, HalfEdge*>* edges);


};


#endif /* HALFEDGEMESH_H_ */
