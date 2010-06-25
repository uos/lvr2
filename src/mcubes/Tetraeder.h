/*
 * Tetraeder.h
 *
 *  Created on: 07.10.2008
 *      Author: twiemann
 */

#ifndef TETRAEDER_H_
#define TETRAEDER_H_

#include "Interpolator.h"
#include "Tables.h"

#include <lib3d/ColorVertex.h>
#include <lib3d/TriangleMesh.h>

class Tetraeder {
public:

	Tetraeder();
	Tetraeder(Vertex v[]);
	virtual ~Tetraeder();

	int getApproximation(int globalIndex, TriangleMesh &mesh, Interpolator* df);

private:
	void initIntersections();

	ColorVertex calcIntersection(int v1, int v2);
	float calcIntersection(float x1, float x2, float v1, float v2, bool interpolate);

	ColorVertex vertices[4];
	ColorVertex intersections[6];
	float values[4];

};

#endif /* TETRAEDER_H_ */
