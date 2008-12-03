/*
 * FastBox.h
 *
 *  Created on: 22.10.2008
 *      Author: Thomas Wiemann
 */

#ifndef FASTBOX_H_
#define FASTBOX_H_

#include "QueryPoint.h"
#include "Interpolator.h"
#include "Tables.h"

#include "../lib3d/TriangleMesh.h"
#include "../lib3d/HalfEdgeMesh.h"

class FastBox {
public:
	FastBox();
	FastBox(const FastBox &other);

	int calcApproximation(vector<QueryPoint> &qp, TriangleMesh &mesh, int global_index);
	int calcApproximation(vector<QueryPoint> &qp, HalfEdgeMesh &mesh, int global_index);
	//int calcApproximationHE(vector<QueryPoint> &qp, HalfEdgeMesh &mesh, int global_index);
	HalfEdge* halfEdgeToVertex(HalfEdgeVertex* v, HalfEdgeVertex* next);

	virtual ~FastBox();

	int vertices      [8];
	int intersections [12];

	FastBox* neighbors[27];

	bool configuration[8];

	int getIndex() const;

	void getCorners(ColorVertex corners[], vector<QueryPoint> &qp);
	void getDistances(float distances[], vector<QueryPoint> &qp);
	void getIntersections(ColorVertex corners[], float distances[], ColorVertex positions[]);

	float calcIntersection(float x1, float x2, float v1, float v2, bool interpolate);

	static int neighbor_table[12][3];
	static int neighbor_vertex_table[12][3];

};

#endif /* FASTBOX_H_ */
