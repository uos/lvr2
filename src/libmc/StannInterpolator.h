/*
 * StannInterpolator.h
 *
 *  Created on: 29.10.2008
 *      Author: Thomas Wiemann
 */

#ifndef STANNINTERPOLATOR_H_
#define STANNINTERPOLATOR_H_

#include <omp.h>

#include "Interpolator.h"

#include "../ann/ANN/ANN.h"

#include "../newmat/newmat.h"
#include "../newmat/newmatap.h"

#include "../lib3d/ColorVertex.h"
#include "../lib3d/Normal.h"

#include "../stann/sfcnn.hpp"

#include "PlaneInterpolator.h"

class StannInterpolator: public Interpolator {
public:
	StannInterpolator(ANNpointArray points, int n, float voxelsize, int k_max, float epsilon);

	virtual float distance(ColorVertex v);

	virtual ~StannInterpolator();

private:

	void estimate_normals();
	void interpolateNormals(int k);
	bool boundingBoxOK(double dx, double dy, double dz);

	Plane calcPlane(Vertex query_point, int k, vector<unsigned long> id);
	Vertex fromID(int i){ return Vertex(points[i][0], points[i][1], points[i][2]);};

	float distance(Vertex v, Plane p);
	float meanDistance(Plane p, vector<unsigned long>, int k);

	ANNpointArray points;
	sfcnn< ANNpoint, 3, double> point_tree;

	float voxelsize;
	float vs_sq;
	float epsilon;

	vector<Normal> normals;

	int number_of_points;
	int k_max;
};

#endif /* STANNINTERPOLATOR_H_ */
