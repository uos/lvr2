/*
 * NormalVoting.h
 *
 *  Created on: 05.03.2009
 *      Author: twiemann
 */

#ifndef NORMALVOTING_H_
#define NORMALVOTING_H_

#include <string.h>

#include <ANN/ANN.h>
#include "StannInterpolator.h"

#include "NormalBucket.h"

class NormalVoting {

public:
	NormalVoting(string filename, float voxelsize);

	void readPoints(string filename);
	void vote();
	void save(string filename);

	virtual ~NormalVoting();

private:

	int getFieldsPerLine(string s);

	StannInterpolator* interpolator;
	ANNpointArray points;

	int number_of_points;
	int biggest_bucket;

	float voxelsize;

	vector<NormalBucket> buckets;

};

#endif /* NORMALVOTING_H_ */
