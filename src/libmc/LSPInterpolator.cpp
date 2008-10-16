/*
 * LSPInterpolator.cpp
 *
 *  Created on: 09.10.2008
 *      Author: twiemann
 */

#include "LSPInterpolator.h"

LSPInterpolator::LSPInterpolator(ANNpointArray pts, int n, float vs, int km, float epsilon) {

	k_max = km;
	points = pts;
	number_of_points = n;
	voxelsize = vs;
	vs_sq = vs * vs;

	cout << "##### Creating ANN Kd-Tree..." << endl;
	ANNsplitRule split = ANN_KD_SUGGEST;
	point_tree = new ANNkd_tree(points, number_of_points, 3, 10, split);

	interpolateNormals();
	write_normals();

}

LSPInterpolator::~LSPInterpolator() {
	// TODO Auto-generated destructor stub
}

void LSPInterpolator::interpolateNormals(){

	int k = 50;

	ANNidxArray id = new ANNidx[k];
	ANNdistArray di = new ANNdist[k];

	Vertex current_point;
	vector<Vertex> pts;

	//for(int i = 0; i < number_of_points; i++) pts.push_back(Vertex(points[i][0], points[i][1], points[i][2]));

	for(int i = 0; i < number_of_points; i++){
		if(i % 100 == 0) cout << "##### LSP: Estimating Normals... " << i << " / " << number_of_points << endl;

		current_point = Vertex(points[i][0], points[i][1], points[i][2]);

		point_tree->annkSearch(points[i], k, id, di);

		pts.clear();
		for(int j = 0; j < k; j++) pts.push_back(Vertex(points[id[j]][0], points[id[j]][1], points[id[j]][2]));

		Normal normal;
		Vertex projection;

		pointLSP(current_point, pts, k, projection, normal);

		normals.push_back(normal);
	}


}

void LSPInterpolator::pointLSP(Vertex p, vector<Vertex> pts, int k, Vertex &projection, Vertex &normal){

	int n = 0;
	int max_iterations = 11;

	vector<Vertex> cpy = pts;
	vector<Vertex> tmp;
	vector<float> a;
	float t = 0.0;
	float t_s;
	float epsilon = 0.0000001;

	float a_max, a_mean, a_limit;

	while(n++ < max_iterations){

		optimalProjection(p, cpy, normal, projection, t_s, a);
		if(fabs(t - t_s) < epsilon) return;

		a_max = -1e10;
		float sum = 0.0;
		for(size_t i = 0; i < a.size(); i++){
			a_max = max(a_max, a[i]);
			sum += a[i];
		}
		a_mean = sum / a.size();

		if(n < 11)
			a_limit = a_mean + ((a_max - a_mean) / (12 - n));
		else
			a_limit = a_mean + ((a_max - a_mean) / 2);

		tmp.clear();
		for(size_t i = 0; i < tmp.size(); i++){
			if(a[i] >= a_limit) tmp.push_back(cpy[i]);
		}
		cpy = tmp;
		if(cpy.size() == 0) return;

	}

}
void LSPInterpolator::optimalProjection(Vertex test_point, vector<Vertex> pts, Vertex &normal,
		               Vertex &projection, float &t, vector<float> &a){

	for(size_t i = 0; i < pts.size(); i++){
		Vertex diff = test_point - pts[i];
		float d = diff.length();
		float a_i = d * d * d * d;
		a.push_back(a_i);
	}

	float c[4] = {0.0, 0.0, 0.0, 0.0};
	for(size_t i = 0; i < pts.size(); i++){
		c[0] += a[i];
		c[1] += a[i] * pts[i].x;
		c[2] += a[i] * pts[i].y;
		c[3] += a[i] * pts[i].z;
	}

	float m[3];
	m[0] = (c[1] / c[0]) - test_point.x;
	m[1] = (c[2] / c[0]) - test_point.y;
	m[2] = (c[3] / c[0]) - test_point.z;

	normal = Normal(-0.5 * m[0],
			        -0.5 * m[1],
			        -0.5 * m[2]);

	float beta = (c[1] * normal.x + c[2] * normal.y + c[3] * normal.z) / c[0];

	t = beta - test_point * normal;

	cout << "T: " << t << endl;

	Vertex diff(t * normal.x, t * normal.y, t * normal.z);
	projection = test_point + diff;

}

void LSPInterpolator::write_normals(){

	ofstream out("normals.nor");
	for(int i = 0; i < number_of_points; i++){
		if(i % 10000 == 0) cout << "##### Writing points and normals: " << i
		                       << " / " << number_of_points << endl;
		out << points[i][0] << " " << points[i][1] << " " << points[i][2] << " "
		    << normals[i][0] << " " << normals[i][1] << " " << normals[i][2] << endl;
	}

}


float LSPInterpolator::distance(ColorVertex v){

	return 0;

}
