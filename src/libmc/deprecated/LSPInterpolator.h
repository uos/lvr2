/*
 * LSPInterpolator.h
 *
 *  Created on: 09.10.2008
 *      Author: twiemann
 */

#ifndef LSPINTERPOLATOR_H_
#define LSPINTERPOLATOR_H_

#include <ANN/ANN.h>

#include <vector>
#include <fstream>

using namespace std;

#include "../newmat/newmat.h"
#include "../newmat/newmatap.h"

#include "../lib3d/ColorVertex.h"
#include "../lib3d/Normal.h"

#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_matrix.h>

#include "Interpolator.h"

class LSPInterpolator : public Interpolator {
public:
	LSPInterpolator(ANNpointArray points, int n, float voxelsize, int k_max, float epsilon);
	virtual ~LSPInterpolator();

	float distance(ColorVertex v);

private:

	void pointLSP(Vertex p, vector<Vertex> pts, int k, Vertex &projection, Vertex &normal);
	void optimalProjection(Vertex test_point, vector<Vertex> pts, Vertex &normal,
			               Vertex &projection, float &t, vector<float> &a);

	void interpolateNormals();
	void write_normals();

	ANNkd_tree* point_tree;
	ANNpointArray points;
	float voxelsize;
	float vs_sq;
	float epsilon;

	int number_of_points;
	int k_max;

	vector<Normal> normals;
};

#endif /* LSPINTERPOLATOR_H_ */
