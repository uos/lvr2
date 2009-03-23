/*
 * RansacInterpolator.cpp
 *
 *  Created on: 28.10.2008
 *      Author: twiemann
 */

#include "RansacInterpolator.h"

RansacInterpolator::RansacInterpolator(ANNpointArray pts, int n, float vs, int km, float epsilon) {

	k_max = km;
	points = pts;
	number_of_points = n;
	voxelsize = vs;
	vs_sq = vs * vs;

	cout << "##### Creating ANN Kd-Tree..." << endl;
	ANNsplitRule split = ANN_KD_SUGGEST;
	point_tree = new ANNkd_tree(points, number_of_points, 3, 10, split);

	srand(time(0));
}



RansacInterpolator::~RansacInterpolator() {
	// TODO Auto-generated destructor stub
}

int RansacInterpolator::random(int k_max){
	return rand() % (number_of_points - 1);
}

float RansacInterpolator::distanceFromPlane(ColumnVector C, int index){
	float y_plane = C(1) + C(2) * points[index][0] + C(3) * points[index][2];
	return fabs(points[index][1] - y_plane);
}

Normal RansacInterpolator::calcNormal(ColumnVector C){
	Normal normal;
	Vertex diff1, diff2;
	Vertex query_point(0.0, 0.0, 0.0);
	float z1, z2;
	float epsilon = 20.0;


	z1 = C(1) + C(2) * (query_point.x + epsilon) + C(3) * query_point.z;
	z2 = C(1) + C(2) * query_point.x + C(3) * (query_point.z + epsilon);

	diff1 = BaseVertex(query_point.x + epsilon, z1, query_point.z) - query_point;
	diff2 = BaseVertex(query_point.x, z2, query_point.z + epsilon) - query_point;

	normal = diff1.cross(diff2);

	normal.normalize();
	return normal;
}

ColumnVector RansacInterpolator::fitPlane(ANNidxArray id, int k){

	ColumnVector C(3);
	try{
		ColumnVector F(k);
		Matrix B(k, 3);

		for(int j = 1; j <= k; j++){
			F(j) = points[id[j-1]][1];
			B(j, 1) = 1;
			B(j, 2) = points[id[j-1]][0];
			B(j, 3) = points[id[j-1]][2];
		}

		Matrix Bt = B.t();
		Matrix BtB = Bt * B;
		Matrix BtBinv = BtB.i();
		Matrix M = BtBinv * Bt;
		C = M * F;

	} catch (Exception e){
		C(1) = 0.0;
		C(2) = 0.0;
		C(3) = 0.0;
	}

	return C;
}

void RansacInterpolator::selectRandomPoints(ANNidxArray indices){

	int id1 = -1;
	int id2 = -1;
	int id3 = -1;

	id1 = random(number_of_points);
	//cout << "ID1: " << id1 << endl;

	do{
		id2 = random(number_of_points);
		//cout << "ID2: " << id1 << " " << id2 << " " << number_of_points << endl;
	} while(id2 == id1);

	do{
		id3 = random(number_of_points);
	}while((id3 == id1) || (id3 == id2));

	indices[0] = id1;
	indices[1] = id2;
	indices[2] = id3;

}

float RansacInterpolator::distance(ColorVertex v){

	epsilon = 15;

	int k = 20;
	int max_iterations = 4;

	vector<vector<int> > c_sets;

	ANNidxArray id = 0;
	ANNdistArray di = 0;

	//Find k nearest points
	id = new ANNidx[k];
	di = new ANNdist[k];

	ANNpoint p = annAllocPt(3);
	p[0] = v.x; p[1] = v.y; p[2] = v.z;
	point_tree->annkSearch(p, k, id, di);

	ANNidxArray indices = new ANNidx[3];
	vector<int> con_set;
	for(int i = 0; i < max_iterations; i++){
		con_set.clear();
		//Select 3 random indices
		selectRandomPoints(indices);

		//Fit plane to these points
		ColumnVector C = fitPlane(indices, 3);

		//Filter points
		for(int j = 0; j < k; j++){
			if(distanceFromPlane(C, id[j]) < epsilon) con_set.push_back(id[j]);
		}
		c_sets.push_back(con_set);
	}

	//Find biggest subset
	size_t max_size = 0;
	int max_index = 0;
	for(size_t i = 0; i < c_sets.size(); i++){
		//cout << i << " " << c_sets[i].size() << endl;
		if(c_sets[i].size() > max_size){
			max_size = c_sets[i].size();
			max_index = i;
		}
	}

	//if(c_sets[max_index].size() == 0) cout << "ERROR" << endl;

	//Create new index array with points in subset

	int n = (int)c_sets[max_index].size();
	ANNidxArray ransac_id;
	ransac_id = new ANNidx[n];
	for(int i = 0; i < n; i++){
		ransac_id[i] = c_sets[max_index][i];
	}

	ColumnVector C = fitPlane(ransac_id, c_sets[max_index].size() );

	//Calculate Normal
	Normal normal = calcNormal(C);
	if(normal * v < 0) normal = normal * 1;

	Vertex nearest(points[id[0]][0], points[id[0]][1], points[id[0]][2]);

	float distance = (v - nearest) * normal;

	delete[] ransac_id;
	delete[] id;
	delete[] di;

	return distance;
}
