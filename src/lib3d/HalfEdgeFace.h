/*
 * HalfEdgeFace.h
 *
 *  Created on: 03.12.2008
 *      Author: Thomas Wiemann
 */

#ifndef HALFEDGEFACE_H_
#define HALFEDGEFACE_H_

#include <vector>
using namespace std;

#include "Normal.h"
#include "HalfEdge.h"
#include "HalfEdgeVertex.h"

class HalfEdgeFace{
public:
	HalfEdgeFace();
	HalfEdgeFace(const HalfEdgeFace &o);

	void calc_normal();
	void interpolate_normal();

	void getVertexNormals(vector<Normal> &n);
	void getVertices(vector<Vertex> &v);
	void getAdjacentFaces(vector<HalfEdgeFace*> &adj);

	Normal getFaceNormal();
	Normal getInterpolatedNormal();

	Vertex getCentroid();

	HalfEdge* edge;
	bool used;
	vector<int> indices;
	int index[3];
	int mcIndex;
	int texture_index;

	//unsigned int face_index;

	unsigned int face_index;

	Normal normal;
};


#endif /* HALFEDGEFACE_H_ */
