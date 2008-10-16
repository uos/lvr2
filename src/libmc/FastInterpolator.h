/*
 * FastInterpolator.h
 *
 *  Created on: 01.10.2008
 *      Author: twiemann
 */

#ifndef FASTINTERPOLATOR_H_
#define FASTINTERPOLATOR_H_

#include "interpolator.h"

#include <ANN/ANN.h>

#include <vector>
#include <fstream>
using namespace std;

#include "../newmat/newmat.h"
#include "../mesh/colorVertex.h"
#include "../mesh/normal.h"

#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_matrix.h>

class FastInterpolator : public Interpolator{
public:
	FastInterpolator(ANNpointArray pts, int n, float vs, int km, float epsilon);
	float distance(ColorVertex v);
	virtual ~FastInterpolator();

private:

	void calcTangentPlanes(int k);
	void interpolateNormals(int k);
	void writeNormals();

	Normal pca(ANNidxArray id, int k, Vertex centroid);

	ANNkd_tree* point_tree;
	ANNkd_tree* centroid_tree;
	ANNkd_tree* normal_tree;

	ANNpointArray points;
	ANNpointArray centroids;
	ANNpointArray normals;

	float voxelsize;
	float vs_sq;
	float epsilon;

	int number_of_points;
	int k_max;

	Vertex offset;
};

#endif /* FASTINTERPOLATOR_H_ */
