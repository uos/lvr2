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
#include <set>

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
	void write_face_normals(string filename);

	void printStats();
	void check_next_neighbor(HalfEdgeFace* f0, HalfEdgeFace* face, HalfEdgePolygon*);

	void extract_borders();
	void generate_polygons();

	void getArea(set<HalfEdgeFace*> &faces, HalfEdgeFace* face, int depth, int max);
	void shiftIntoPlane(HalfEdgeFace* f);

	bool check_face(HalfEdgeFace* f0, HalfEdgeFace* current);
	bool isFlatFace(HalfEdgeFace* face);

	void create_polygon(vector<int> &polygon, hash_map<unsigned int, HalfEdge*>* edges);

private:

	int biggest_size;
	HalfEdgePolygon* biggest_polygon;

	float  current_d;
	Normal current_n;
	Vertex current_v;

};


#endif /* HALFEDGEMESH_H_ */
