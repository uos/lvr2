/*
 * PlaneInterpolator.cpp
 *
 *  Created on: 07.10.2008
 *      Author: twiemann
 */

#include "PlaneInterpolator.h"

PlaneInterpolator::PlaneInterpolator(ANNpointArray pts, int n, float vs, int km, float epsilon) {

	k_max = km;
	points = pts;
	number_of_points = n;
	voxelsize = vs;
	vs_sq = vs * vs;

	cout << "##### Creating ANN Kd-Tree..." << endl;
	ANNsplitRule split = ANN_KD_SUGGEST;
	point_tree = new ANNkd_tree(points, number_of_points, 3, 10, split);

	estimate_normals();
	write_normals();

}

void PlaneInterpolator::write_normals(){

	ofstream out("normals.nor");
	for(int i = 0; i < number_of_points; i++){
		if(i % 10000 == 0) cout << "##### Writing points and normals: " << i
		                       << " / " << number_of_points << endl;
		out << points[i][0] << " " << points[i][1] << " " << points[i][2] << " "
		    << normals[i][0] << " " << normals[i][1] << " " << normals[i][2] << endl;
	}

}

float PlaneInterpolator::y(Plane p, float x, float y){

	return p.a  + p.b * x + p.c * y;

}

float PlaneInterpolator::distance(Vertex v, Plane p){
	return fabs((v - p.p) * p.n);
}

float PlaneInterpolator::meanDistance(Plane p, ANNidxArray id, int k){

	float sum = 0.0;
	for(int i = 0; i < k; i++){
		sum += distance(fromID(id[i]), p);
	}
	sum = sum / k;
	return sum;

}

bool PlaneInterpolator::boundingBoxOK(double dx, double dy, double dz){

	float e = 0.05;

	if(dx < e * dy) return false;
	else if(dx < e * dz) return false;
	else if(dy < e * dx) return false;
	else if(dy < e * dz) return false;
	else if(dz < e * dx) return false;
	else if(dy < e * dy) return false;

	return true;
}

void PlaneInterpolator::estimate_normals(){

	Vertex query_point, diff1, diff2;
	Normal normal;

	ANNidxArray id = 0;
	ANNdistArray di = 0;

	float epsilon = 1.0;

	int k_0 = 50;
	int k = 0;
	int n = 0;

	float mean_distance;

	cout << "##### Initializing normal array..." << endl;
	//Initialize normals
	for(int i = 0; i < number_of_points; i++) normals.push_back(Normal(0.0, 0.0, 0.0));

	for(int i = 0; i < number_of_points; i++){

		n = 0;
		k = k_0;

		if(i % 10000 == 0) cout << "##### Estimating Normals... " << i << " / " << number_of_points << endl;

		while(n < 5){

			n++;
			k = k * 2;

			if(di != 0) delete[] di;
			if(id != 0) delete[] id;

			id = new ANNidx[k];
			di = new ANNdist[k];

			point_tree->annkSearch(points[i], k, id, di);

			double min_x = 1e15;
			double min_y = 1e15;
			double min_z = 1e15;
			double max_x = - min_x;
			double max_y = - min_y;
			double max_z = - min_z;

			double dx, dy, dz;

			for(int j = 0; j < k; j++){

				min_x = min(min_x, points[id[j]][0]);
				min_y = min(min_y, points[id[j]][1]);
				min_z = min(min_z, points[id[j]][2]);

				max_x = max(max_x, points[id[j]][0]);
				max_y = max(max_y, points[id[j]][1]);
				max_z = max(max_z, points[id[j]][2]);

				dx = max_x - min_x;
				dy = max_y - min_y;
				dz = max_z - min_z;

			}

			if(boundingBoxOK(dx, dy, dz)) break;

 		}

		//cout << n << " " << k << endl;

		query_point = Vertex(points[i][0], points[i][1], points[i][2]);

		Plane p = calcPlane(query_point, k, id);

		mean_distance = meanDistance(p, id, k);
		//cout << mean_distance <<  endl;

		for(int j = 0; j < k; j++){
			float dst= distance(fromID(id[j]), p);
			if(dst < epsilon * mean_distance){
				Normal n = normals[id[j]] + p.n;
				if(n * query_point < 0) n = n * -1;
				normals[id[j]] = n;
			}
		}


//		Normal normal =  p.n;
//		if(normal * query_point < 0) normal = normal * -1;
//		normals[i] = normal;

	}

	cout << endl;

	delete[] id;
	delete[] di;

	interpolateNormals(20);
}

Plane PlaneInterpolator::calcPlane(Vertex query_point, int k, ANNidxArray id){

	Vertex diff1, diff2;
	Normal normal;

	ColumnVector C(3);
	float z1, z2;

	epsilon = 20.0;

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

		z1 = C(1) + C(2) * (query_point.x + epsilon) + C(3) * query_point.z;
		z2 = C(1) + C(2) * query_point.x + C(3) * (query_point.z + epsilon);

		//cout << z1 << " " << z2 << " " << epsilon << endl;

		diff1 = BaseVertex(query_point.x + epsilon, z1, query_point.z) - query_point;
		diff2 = BaseVertex(query_point.x, z2, query_point.z + epsilon) - query_point;

		normal = diff1.cross(diff2);

	} catch (Exception e){
		normal = Normal(0.0, 0.0, 0.0);
	}

	Plane p;
	p.a = C(1);
	p.b = C(2);
	p.c = C(3);
	p.n = normal;
	p.p = query_point;

	return p;
}


void PlaneInterpolator::interpolateNormals(int k){

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
				normals[id[j]] = mean_normal;
			}
		}

	}

	cout << "##### Copying normals..." << endl;

	for(int i = 0; i < number_of_points; i++){


		normals[i] = tmp[i];
	}

	delete[] id;
	delete[] di;


}

Normal PlaneInterpolator::pca(ANNidxArray id, int k, Vertex centroid){

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
		//k_nearest = points[i];
		k_nearest = Vertex(fabs(points[id[i]][0]), fabs(points[id[i]][1]), fabs(points[id[i]][2]));
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

float PlaneInterpolator::distance(ColorVertex v){

	int k = 40;

	//Allocate ANN point
	ANNpoint p = annAllocPt(3);
	p[0] = v.x; p[1] = v.y; p[2] = v.z;

	//Arrays for indices
	ANNidxArray id = new ANNidx[k];
	ANNdistArray di = new ANNdist[k];

	//Find nearest tangent plane
	point_tree->annkSearch(p, k, id, di);

	Vertex nearest(0.0, 0.0, 0.0);
	Normal normal(0.0, 0.0, 0.0);

	for(int i = 0; i < k; i++){
		//Get nearest tangent plane
		Vertex vq (points[id[i]][0], points[id[i]][1], points[id[i]][2]);

		//Get normal
		Normal n(normals[id[i]][0], normals[id[i]][1], normals[id[i]][2]);

		nearest += vq;
		normal += n;

	}

	normal.x = normal.x / k;
	normal.y = normal.y / k;
	normal.z = normal.z / k;

	nearest.x = nearest.x / k;
	nearest.y = nearest.y / k;
	nearest.z = nearest.z / k;

	//Calculate distance
	float distance = (v - nearest) * normal;

	//De-alloc ANN point
	annDeallocPt(p);

	delete[] di;
	delete[] id;

	return distance;

}

PlaneInterpolator::~PlaneInterpolator() {

}
