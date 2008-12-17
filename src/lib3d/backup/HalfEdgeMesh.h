/*
 * HalfEdgeMesh.h
 *
 *  Created on: 13.11.2008
 *      Author: twiemann
 */

#ifndef HALFEDGEMESH_H_
#define HALFEDGEMESH_H_

#include <vector>
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

	int index;

	vector<HalfEdge*> in;
	vector<HalfEdge*> out;
};

class HalfEdgeFace{
public:
	HalfEdgeFace();
	HalfEdgeFace(const HalfEdgeFace &o);

	void calc_normal();

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
	void check_next_neighbor(float d_0, HalfEdgeFace* face, HalfEdgePolygon* polygon);
	void extract_borders();

	void analize();

	inline int freeman3D(HalfEdge* edge);
	inline int classifyFace(HalfEdgeFace*, float);


};


int HalfEdgeMesh::classifyFace(HalfEdgeFace* face, float d_0){
//  int index = face->mcIndex;
//
//  //WALL
//  if(index == 240 || index == 15 || index == 153 || index == 102){
//
//    return 1;
//
//  }
//  //FLOOR
//  else if(index == 204){
//
//    return 4;
//
//  }
//  //CEIL
//  else if (index == 51){
//
//    return 2;
//
//  }
//  //DOORS
//  else if (index == 9 || index == 144 || index == 96 || index == 6){
//
//    return 3;
//
//  }
//  //OTHER FLAT POLYGONS
//  else if(index ==  68 || index == 136 || index ==  17 || index ==  34 || //Variants of MC-Case 2
//	  index == 192 || index ==  48 || index ==  12 || index ==   3 ){
//
//    return 0;
//
//  } else if (index ==  63 || index == 159 || index == 207 || index == 111 || //Variants of MC-Case 2 (compl)
//		   index == 243 || index == 249 || index == 252 || index == 246 ||
//		   index == 119 || index == 187 || index == 221 || index == 238){
//    return 0;
//
//  }
//  return -1;

	//Calculate center of gravity
	const float epsilon = 0.5;

	Vertex cog;
	for(int i = 0; i < 3; i++) cog += he_vertices[face->index[i]]->position;

	cog.x = cog.x / 3.0;
	cog.y = cog.y / 3.0;
	cog.z = cog.z / 3.0;

	float d = face->normal * cog;

	if(fabs(d_0 - d) < epsilon)
		return 1;
	else
		return -1;


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
