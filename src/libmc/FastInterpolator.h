/*
 * FastInterpolator.h
 *
 *  Created on: 01.10.2008
 *      Author: twiemann
 */

#ifndef FASTINTERPOLATOR_H_
#define FASTINTERPOLATOR_H_

#include "Interpolator.h"

#include <ANN/ANN.h>

#include <vector>
#include <fstream>
using namespace std;

#include "../newmat/newmat.h"

#include "../lib3d/ColorVertex.h"
#include "../lib3d/Normal.h"

#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_matrix.h>

class FastInterpolator : public Interpolator{
public:
	FastInterpolator(ANNpointArray pts, int n, float vs, int km, float epsilon);
	float distance(ColorVertex v);
	virtual ~FastInterpolator();

private:

	Normal calcNormal(Vertex v, int k);
	ColumnVector fitPlane(ANNidxArray id, int k);

	ANNkd_tree* point_tree;
	ANNpointArray points;

	float voxelsize;
	float vs_sq;
	float epsilon;

	int number_of_points;
	int k_max;

	Vertex offset;
};

#endif /* FASTINTERPOLATOR_H_ */
