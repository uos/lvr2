/*
 * PlaneInterpolator.h
 *
 *  Created on: 07.10.2008
 *      Author: twiemann
 */

#ifndef PLANEINTERPOLATOR_H_
#define PLANEINTERPOLATOR_H_

#include <ANN/ANN.h>

#include <vector>
#include <fstream>

using namespace std;

#include "../newmat/newmat.h"
#include "../newmat/newmatap.h"

#include "../mesh/colorVertex.h"
#include "../mesh/normal.h"

#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_matrix.h>

#include "interpolator.h"

struct Plane{
	float a, b, c;
	Normal n;
	Vertex p;
};

class PlaneInterpolator : public Interpolator{
public:
	PlaneInterpolator(ANNpointArray points, int n, float voxelsize, int k_max, float epsilon);
	virtual ~PlaneInterpolator();

	float distance(ColorVertex v);

private:

	Plane calcPlane(Vertex query_point, int k, ANNidxArray id);

	void estimate_normals();
	void interpolateNormals(int k);
	void write_normals();

	bool boundingBoxOK(double dx, double dy, double dz);

	float y(Plane p, float x, float y);
	float meanDistance(Plane p, ANNidxArray id, int k);
	float distance(Vertex v, Plane p);

	Vertex fromID(int i){ return Vertex(points[i][0], points[i][1], points[i][2]);};

	Normal pca(ANNidxArray id, int k, Vertex centroid);

	vector<Normal> normals;

	ANNkd_tree* point_tree;
	ANNpointArray points;
	float voxelsize;
	float vs_sq;
	float epsilon;

	int number_of_points;
	int k_max;
};

#endif /* PLANEINTERPOLATOR_H_ */
