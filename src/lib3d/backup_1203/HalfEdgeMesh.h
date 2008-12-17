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
#include "HalfEdgePolygon.h"

class HalfEdgeVertex;
class HalfEdgeFace;
class HalfEdgePolygon;

class HePlane{

public:
	HePlane(Normal n, Vertex v);
	HePlane(const HePlane &o);

	void interpolate(HePlane p);
	float distance(Vertex p);

	float d;
	Normal n;
};


class HalfEdge{
public:
	HalfEdge();
	~HalfEdge();

	HalfEdge* next;
	HalfEdge* pair;

	HalfEdgeVertex* start;
	HalfEdgeVertex* end;

	HalfEdgeFace* face;

	bool used;
};

class HalfEdgeVertex{
public:
	HalfEdgeVertex();
	HalfEdgeVertex(const HalfEdgeVertex& o);

	Vertex position;
	Normal normal;

	float color;

	int index;

	vector<HalfEdge*> in;
	vector<HalfEdge*> out;
};

class HalfEdgeFace{
public:
	HalfEdgeFace();
	HalfEdgeFace(const HalfEdgeFace &o);

	void calc_normal();
	void interpolate_normal();

	HalfEdge* edge;
	bool used;
	vector<int> indices;
	int index[3];
	int mcIndex;
	int texture_index;

	Normal normal;
};


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
	void check_next_neighbor(HalfEdgeFace* f0, HePlane &p, HalfEdgeFace* face, hash_map<unsigned int, HalfEdge*>* points);
	void extract_borders();

	inline int freeman3D(HalfEdge* edge);
	inline int classifyFace(HalfEdgeFace* f0, HalfEdgeFace*, HePlane &p);

	void create_polygon(vector<int> &polygon, hash_map<unsigned int, HalfEdge*>* edges);


};


int HalfEdgeMesh::classifyFace(HalfEdgeFace* f0, HalfEdgeFace* face, HePlane &p){

	//SEMI WORKING CODE:.................

//	int classification = -1;
//	int index = face->mcIndex;
//
//	if(index == start_index){
//		//WALL
//		if(index == 240 || index == 15 || index == 153 || index == 102){
//			classification = 1;
//		}
//		//FLOOR
//		else if(index == 204){
//			classification =  4;
//		}
//		//CEIL
//		else if (index == 51){
//			classification = 2;
//		}
//		//DOORS
//		else if (index == 9 || index == 144 || index == 96 || index == 6){
//			classification = 3;
//		}
//		//OTHER FLAT POLYGONS
//		else if(index ==  68 || index == 136 || index ==  17 || index ==  34 || //Variants of MC-Case 2
//				index == 192 || index ==  48 || index ==  12 || index ==   3 ){
//
//			classification = 0;
//
//		} else if (index ==  63 || index == 159 || index == 207 || index == 111 || //Variants of MC-Case 2 (compl)
//				index == 243 || index == 249 || index == 252 || index == 246 ||
//				index == 119 || index == 187 || index == 221 || index == 238){
//			classification = 0;
//		}
//	} else {
////		const float deg     = 35;
////		const float rad     = deg * 3.1415926 / 180.0;
////		const float epsilon = cos(rad);
////
////		if(face->normal * p.n > epsilon) classification = 5;
//
//
//		const float epsilon = 10;
//
//		HalfEdgeVertex* v1 = he_vertices[face->index[0]];
//		HalfEdgeVertex* v2 = he_vertices[face->index[1]];
//		HalfEdgeVertex* v3 = he_vertices[face->index[2]];
//
//		Vertex center;Hardyscheibe
//		center += v1->position;
//		center += v2->position;
//		center += v3->position;
//
//		center.x /= 3.0;
//		center.y /= 3.0;
//		center.z /= 3.0;
//
//		HePlane plane(face->normal, center);
//
//		if(fabs(p.distance(center)) < epsilon){
//			classification = 1;
//			//cout << "HÄ???" << endl;
//			//p.interpolate(plane);
//		}
//
//		cout << fabs(p.distance(center)) << endl;
//	}

	float classification = -1;

	const float deg     = 10;
	const float rad     = deg * 3.1415926 / 180.0;
	const float epsilon = cos(rad);

	if(f0->normal * face->normal > epsilon){
		classification = 5;
	} else {
		classification = -1;
	}

	return classification;
}

int HalfEdgeMesh::freeman3D(HalfEdge* edge){

	HalfEdgeVertex* start_vertex = edge->start;
	HalfEdgeVertex* end_vertex = edge->end;

	int freeman_code = -1;

	if(end_vertex->position.x > start_vertex->position.x){

		if(end_vertex->position.y > start_vertex->position.y)
			freeman_code = 2;
		else if (end_vertex->position.y == start_vertex->position.y)
			freeman_code = 1;
		else if (end_vertex->position.y < start_vertex->position.y)
			freeman_code = 8;

	} else if(end_vertex->position.x == start_vertex->position.x){

		if(end_vertex->position.y > start_vertex->position.y)
			freeman_code = 3;
		if(end_vertex->position.y < start_vertex->position.y)
			freeman_code = 7;
		if(end_vertex->position.y == start_vertex->position.y)
			freeman_code = 9;

	} else if(end_vertex->position.x < start_vertex->position.x){

		if(end_vertex->position.y > start_vertex->position.y)
			freeman_code = 4;
		if(end_vertex->position.y == start_vertex->position.y)
			freeman_code = 5;
		if(end_vertex->position.y < start_vertex->position.y)
			freeman_code = 6;

	}

	 if(end_vertex->position.z > start_vertex->position.z) freeman_code *= 10;
	 if(end_vertex->position.z < start_vertex->position.z) freeman_code *= 100;

	return freeman_code;
}

#endif /* HALFEDGEMESH_H_ */
