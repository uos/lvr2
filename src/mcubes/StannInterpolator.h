/*
 * StannInterpolator.h
 *
 *  Created on: 29.10.2008
 *      Author: Thomas Wiemann
 */

#ifndef STANNINTERPOLATOR_H_
#define STANNINTERPOLATOR_H_

#include <omp.h>
#include <iostream>
#include <fstream>

using namespace std;

#include "Interpolator.h"

#include "../ann/ANN/ANN.h"

#include "../newmat/newmat.h"
#include "../newmat/newmatap.h"

#include <lib3d/ColorVertex.h>
#include <lib3d/Normal.h>

#include "../stann/sfcnn.hpp"

#include <sys/time.h>
#include <time.h>
#include <unistd.h>

unsigned long GetCurrentTimeInMilliSec(void);


struct Plane{
	float a, b, c;
	Normal n;
	Vertex p;
};

class StannInterpolator: public Interpolator {
public:
	StannInterpolator(
			ANNpointArray points, int n, float voxelsize, int k_max,
			float epsilon, Vertex c = Vertex(0.0, 0.0, 0.0));

	virtual float distance(ColorVertex v);

	virtual ~StannInterpolator();

	Vertex center;

//private:

	void write_normals();
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
