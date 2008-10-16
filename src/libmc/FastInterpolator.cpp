/*
 * FastInterpolator.cpp
 *
 *  Created on: 01.10.2008
 *      Author: twiemann
 */

#include "FastInterpolator.h"

FastInterpolator::FastInterpolator(ANNpointArray pts, int n, float vs, int km, float epsilon) {

	float k = 20;

	k_max = km;
	points = pts;
	number_of_points = n;
	voxelsize = vs;
	vs_sq = 0.5 * vs * vs;

	ANNsplitRule split = ANN_KD_SUGGEST;
	point_tree = new ANNkd_tree(points, number_of_points, 3, 10, split);

	calcTangentPlanes(k);
	writeNormals();

}

void FastInterpolator::calcTangentPlanes(int k){

	ANNidxArray id = new ANNidx[k];
	ANNdistArray di = new ANNdist[k];

	centroids = annAllocPts(number_of_points, 3);
	normals = annAllocPts(number_of_points, 3);

	for(int i = 0; i < number_of_points; i++){

		if(i % 10000 == 0) cout << endl << "##### Calculating tangent planes: "
		                       << i << " / " << number_of_points << " " << flush;
		if(i % 1000  == 0) cout << "." << flush;


		//Find k nearest neighbours
		point_tree->annkSearch(points[i], k, id, di);

		//Calculate centroid
		Vertex centroid;
		for(int j = 0; j < k; j++){
			centroid.x += points[id[j]][0];
			centroid.y += points[id[j]][1];
			centroid.z += points[id[j]][2];
		}
		//centroid /= k;
		centroid.x = centroid.x / k;
		centroid.y = centroid.y / k;
		centroid.z = centroid.z / k;

		//cout << centroid;

		centroids[i][0] = centroid.x;
		centroids[i][1] = centroid.y;
		centroids[i][2] = centroid.z;

		//calculate normal
		Normal normal = pca(id, k, Vertex(1.0, 1.0, 1.0));

		//Flip normal
		Vertex qp(points[i][0], points[i][1], points[i][2]);
		if(normal * qp < 0) normal = normal * -1;


		//Save normal
		normals[i][0] = normal.x;
		normals[i][1] = normal.y;
		normals[i][2] = normal.z;

	}

	cout << endl;

	//Build centroid tree
	ANNsplitRule split = ANN_KD_SUGGEST;
	centroid_tree = new ANNkd_tree(centroids, number_of_points, 3, 10, split);

	interpolateNormals(k);
}

void FastInterpolator::interpolateNormals(int k){
//
//	ANNidxArray id = new ANNidx[k];
//	ANNdistArray di = new ANNdist[k];
//
//	vector<Normal> tmp;
//
//	for(int i = 0; i < number_of_points; i++){
//		point_tree->annkSearch(points[i], k, id, di);
//
//		if(i % 10000 == 0) cout << "##### Interpolating normals: "
//				                << i << " / " << number_of_points << endl;
//
//		Vertex mean;
//		Normal mean_normal;
//
//		for(int j = 0; j < k; j++){
//			mean += Vertex(normals[id[j]][0],
//					       normals[id[j]][1],
//					       normals[id[j]][2]);
//		}
//		mean_normal = Normal(mean);
//
//
//		for(int j = 0; j < k; j++){
//			Normal n = Normal(normals[id[j]][0], normals[id[j]][1], normals[id[j]][2]);
//			if(fabs(n * mean_normal) > 0.2 ){
//				normals[id[j]][0] = mean_normal.x;
//				normals[id[j]][1] = mean_normal.y;
//				normals[id[j]][2] = mean_normal.z;
//			}
//		}
//
//	}
//
//	delete[] id;
//	delete[] di;

	ANNidxArray id = new ANNidx[k];
	ANNdistArray di = new ANNdist[k];

	vector<Normal> tmp;

	for(int i = 0; i < number_of_points; i++){
		point_tree->annkSearch(points[i], k, id, di);

		if(i % 10000 == 0) cout << "##### Interpolating normals: "
		<< i << " / " << number_of_points << endl;

		Vertex mean;
		Normal mean_normal;

		for(int j = 0; j < k; j++){
			mean += Vertex(normals[id[j]][0],
					normals[id[j]][1],
					normals[id[j]][2]);
		}
		mean_normal = Normal(mean);

		tmp.push_back(mean);

		for(int j = 0; j < k; j++){
			Normal n = Normal(normals[id[j]][0], normals[id[j]][1], normals[id[j]][2]);
			if(fabs(n * mean_normal) > 0.2 ){
				normals[id[j]][0] = mean_normal.x;
				normals[id[j]][1] = mean_normal.y;
				normals[id[j]][2] = mean_normal.z;
			}
		}

	}

	cout << "##### Copying normals..." << endl;

	for(int i = 0; i < number_of_points; i++){
		normals[i][0] = tmp[i].x;
		normals[i][1] = tmp[i].y;
		normals[i][2] = tmp[i].z;
	}

	delete[] id;
	delete[] di;


}

Normal FastInterpolator::pca(ANNidxArray id, int k, Vertex centroid){

	Vertex vector_to_centroid, k_nearest;

	//Init covariance matrix
	float matrix[3][3];
	for(int i = 0; i < 3; i++){
		for(int j = 0; j < 3; j++){
			matrix[i][j] = 0.0;
		}
	}

	//Calculate covariance matrix
	for(int i = 0; i < k; i++){
		k_nearest = Vertex(fabs(points[id[i]][0]), fabs(points[id[i]][1]), fabs(points[id[i]][2]));
		//k_nearest = Vertex(fabs(points[id[i]][0]), fabs(points[id[i]][1]), fabs(points[id[i]][2]));
		vector_to_centroid = k_nearest - centroid;
		for(int k = 0; k < 3; k++){
			for(int l = 0; l < 3; l++){
				matrix[k][l] += vector_to_centroid[k] * vector_to_centroid[l];
			}
		}
	}

	//Copy covariance matrix into gsl matrix
	gsl_matrix* mat = gsl_matrix_alloc(3, 3);
	for(int i = 0; i < 3; i++){
		for(int j = 0; j < 3; j++){
			gsl_matrix_set(mat, i, j, matrix[i][j]);
		}
	}


	//Calculate eigenvalues
	gsl_eigen_symmv_workspace * w = gsl_eigen_symmv_alloc(3);
	gsl_vector *eval = gsl_vector_alloc(3);
	gsl_matrix *evec = gsl_matrix_alloc(3, 3);
	gsl_eigen_symmv (mat, eval, evec, w);

	gsl_eigen_symmv_sort (eval, evec, GSL_EIGEN_SORT_ABS_ASC);

	//Get fist eigenvalue != 0
	double i = 0.0;
	int stelle = 0;
	while (i != 0.0 && stelle <= 3) {
		i = gsl_vector_get(eval, stelle);
		stelle++;
	}

	//Set eigenvector
	gsl_vector *re = gsl_vector_alloc(3);
	for (int j = 0; j < 3; j++) {
		gsl_vector_set(re, j, gsl_matrix_get(evec, stelle, j));
	}

	Normal normal(gsl_vector_get(re, 0),
			      gsl_vector_get(re, 1),
			      gsl_vector_get(re, 2));


	gsl_eigen_symmv_free(w);
	gsl_matrix_free(evec);
	gsl_vector_free(eval);
	gsl_vector_free(re);

	return normal;

}

FastInterpolator::~FastInterpolator() {
}

void FastInterpolator::writeNormals(){

	ofstream out("normals.nor");
	for(int i = 0; i < number_of_points; i++){
		if(i % 10000 == 0) cout << "##### Writing points and normals: " << i
		                       << " / " << number_of_points << endl;
		out << points[i][0] << " " << points[i][1] << " " << points[i][2] << " "
		    << normals[i][0] << " " << normals[i][1] << " " << normals[i][2] << endl;
	}

}

float FastInterpolator::distance(ColorVertex v){


	//Allocate ANN point
	ANNpoint p = annAllocPt(3);
	p[0] = v.x; p[1] = v.y; p[2] = v.z;

	//Arrays for indices
	ANNidxArray id = new ANNidx[10];
	ANNdistArray di = new ANNdist[10];

	//Find nearest tangent plane
	centroid_tree->annkSearch(p, 1, id, di);

	//cout << id[0] << endl;

	//Get nearest tangent plane
	Vertex nearest(centroids[id[0]][0],
			       centroids[id[0]][1],
			       centroids[id[0]][2]);

	//Get normal
	Normal normal(normals[id[0]][0],
			      normals[id[0]][1],
			      normals[id[0]][2]);


	//Calculate distance
	float distance = (v - nearest) * normal;

	//De-alloc ANN point
	annDeallocPt(p);

	delete[] di;
	delete[] id;

	return distance;
}
