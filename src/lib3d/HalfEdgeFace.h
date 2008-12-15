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
