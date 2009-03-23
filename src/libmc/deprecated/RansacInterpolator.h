/*
 * RansacInterpolator.h
 *
 *  Created on: 28.10.2008
 *      Author: twiemann
 */

#ifndef RANSACINTERPOLATOR_H_
#define RANSACINTERPOLATOR_H_

#include <stdlib.h>
#include <time.h>

#include <vector>
using namespace std;

#include "../newmat/newmat.h"
#include "../newmat/newmatap.h"

#include "../ann/ANN/ANN.h"

#include "../lib3d/Normal.h"

#include "Interpolator.h"

class RansacInterpolator: public Interpolator {
public:
	RansacInterpolator(ANNpointArray pts, int n, float vs, int km, float epsilon);

	ColumnVector fitPlane(ANNidxArray id, int k);
	Normal       calcNormal(ColumnVector c);

	virtual float distance(ColorVertex v);
	virtual ~RansacInterpolator();

private:

	int random(int k_max);
	float distanceFromPlane(ColumnVector C, int index);

	void selectRandomPoints(int* indices);

	ANNkd_tree* point_tree;
	ANNpointArray points;
	float voxelsize;
	float vs_sq;
	float epsilon;

	int number_of_points;
	int k_max;
};

#endif /* RANSACINTERPOLATOR_H_ */
