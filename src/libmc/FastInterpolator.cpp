/*
 * FastInterpolator.cpp
 *
 *  Created on: 01.10.2008
 *      Author: twiemann
 */

#include "FastInterpolator.h"

FastInterpolator::FastInterpolator(ANNpointArray pts, int n, float vs, int km, float epsilon) {

	k_max = km;
	points = pts;
	number_of_points = n;
	voxelsize = vs;
	vs_sq = 0.5 * vs * vs;

	ANNsplitRule split = ANN_KD_SUGGEST;
	point_tree = new ANNkd_tree(points, number_of_points, 3, 10, split);

}

Normal FastInterpolator::calcNormal(Vertex query_point, int k){

	epsilon = 20;

	//Find k neighbours
	ANNidxArray id = new ANNidx[k];
	ANNdistArray di = new ANNdist[k];

	ANNpoint p = annAllocPt(3);
	p[0] = query_point.x;
	p[1] = query_point.y;
	p[2] = query_point.z;

	point_tree->annkSearch(p, k, id, di);

	//Fit plane to these nb and calc interpolated normal
	ColumnVector C = fitPlane(id, k);

	//Calculate normal
	Normal normal;
	Vertex diff1, diff2;
	float z1, z2;
	float epsilon = 20.0;

	z1 = C(1) + C(2) * (query_point.x + epsilon) + C(3) * query_point.z;
	z2 = C(1) + C(2) * query_point.x + C(3) * (query_point.z + epsilon);

	diff1 = BaseVertex(query_point.x + epsilon, z1, query_point.z) - query_point;
	diff2 = BaseVertex(query_point.x, z2, query_point.z + epsilon) - query_point;

	normal = diff1.cross(diff2);

	delete[] id;
	delete[] di;

	annDeallocPt(p);

	return normal;
}



ColumnVector FastInterpolator::fitPlane(ANNidxArray id, int k){

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


FastInterpolator::~FastInterpolator() {
}



float FastInterpolator::distance(ColorVertex v){

	int k = 20;
	int k_normal_estimation = 10;
	int k_normal_interpolation = 10;

	ANNidxArray id = new ANNidx[k];
	ANNdistArray di = new ANNdist[k];

	ANNidxArray id_n = new ANNidx[k_normal_interpolation];
	ANNdistArray di_n = new ANNdist[k_normal_interpolation];

	ANNpoint p = annAllocPt(3);
	p[0] = v.x;
	p[1] = v.y;
	p[2] = v.z;

	point_tree->annkSearch(p, k, id, di);

	Normal normal;
	Vertex nearest(0.0, 0.0, 0.0);
	for(int i = 0; i < k; i++){

		Normal tmp_normal;
		point_tree->annkSearch(points[i], k_normal_interpolation, id_n, di_n);
		for(int j = 0; j < k_normal_interpolation; j++){
			Normal n = calcNormal(Vertex(points[id_n[j]][0],
					                     points[id_n[j]][1],
					                     points[id_n[j]][2]), k_normal_estimation);

			tmp_normal += n;
		}

		if(tmp_normal * v < 0 ) tmp_normal = tmp_normal * -1;

		tmp_normal.x = tmp_normal.x / k_normal_interpolation;
		tmp_normal.y = tmp_normal.y / k_normal_interpolation;
		tmp_normal.z = tmp_normal.z / k_normal_interpolation;

		normal += tmp_normal;
		nearest += Vertex(points[id[i]][0], points[id[i]][1], points[id[i]][2]);

	}
	normal.x = normal.x / k;
	normal.y = normal.y / k;
	normal.z = normal.z / k;

	nearest.x = nearest.x / k;
	nearest.y = nearest.y / k;
	nearest.z = nearest.z / k;

	float distance = (v - nearest) * normal;

	return distance;

}
