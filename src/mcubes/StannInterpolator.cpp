/*
 * StannInterpolator.cpp
 *
 *  Created on: 29.10.2008
 *      Author: Thomas Wiemann
 */

#include "StannInterpolator.h"

#include "../newmat/newmatio.h"

#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include <boost/progress.hpp>

unsigned long GetCurrentTimeInMilliSec(void)
{
  static struct timeval tv;
  static unsigned long milliseconds;
  gettimeofday(&tv, NULL);
  milliseconds = tv.tv_sec * 1000 + tv.tv_usec / 1000;
  return milliseconds;
}

StannInterpolator::StannInterpolator(float** pts, int n, float vs, int km, float epsilon, Vertex c) {

	center = c;

	cout << center;

	k_max = km;
	points = pts;
	number_of_points = n;
	voxelsize = vs;
	vs_sq = vs * vs;

	normals = 0;

	omp_set_num_threads(4);

	cout << "##### Creating STANN Kd-Tree..." << endl;
	point_tree = sfcnn< float*, 3, float>(points, number_of_points, 4);

	unsigned long start_time = GetCurrentTimeInMilliSec();

	estimate_normals();
	interpolateNormals(20);
	//write_normals();

	unsigned long end_time = GetCurrentTimeInMilliSec();

	cout << "Elapsed time: " << (double)(end_time - start_time) * 0.001 << endl;

}

StannInterpolator::~StannInterpolator() {
}

float StannInterpolator::distance(Vertex v, Plane p){
	return fabs((v - p.p) * p.n);
}

float StannInterpolator::meanDistance(Plane p, vector<unsigned long> id, int k){

	float sum = 0.0;
	for(int i = 0; i < k; i++){
		sum += distance(fromID(id[i]), p);
	}
	sum = sum / k;
	return sum;

}

bool StannInterpolator::boundingBoxOK(double dx, double dy, double dz){

	float e = 0.05;

	if(dx < e * dy) return false;
	else if(dx < e * dz) return false;
	else if(dy < e * dx) return false;
	else if(dy < e * dz) return false;
	else if(dz < e * dx) return false;
	else if(dy < e * dy) return false;

	return true;
}

Plane StannInterpolator::calcPlane(Vertex query_point, int k, vector<unsigned long> id){

	Vertex diff1, diff2;
	Normal normal;

	ColumnVector C(3);
	float z1, z2;

	epsilon = 50000.0;

	try{
		ColumnVector F(k);
		Matrix B(k, 3);

        //#pragma omp parallel for
		for(int j = 1; j <= k; j++){

			F(j) = points[id[j-1]][1];
			B(j, 1) = 1;
			B(j, 2) = points[id[j-1]][0];
			B(j, 3) = points[id[j-1]][2];

		}

		//cout << "Transpose" << endl;

		Matrix Bt = B.t();

		//cout << "Mult1" << endl;

		Matrix BtB = Bt * B;

		//cout << "Invert" << endl;

		Matrix BtBinv = BtB.i();

		//cout << "Mult2" << endl;
		Matrix M = BtBinv * Bt;

		C = M * F;

		//cout << " Coeff" << endl;

		z1 = C(1) + C(2) * (query_point.x + epsilon) + C(3) * query_point.z;
		z2 = C(1) + C(2) * query_point.x + C(3) * (query_point.z + epsilon);

		//cout << z1 << " " << z2 << " " << epsilon << endl;

		diff1 = BaseVertex(query_point.x + epsilon, z1, query_point.z) - query_point;
		diff2 = BaseVertex(query_point.x, z2, query_point.z + epsilon) - query_point;

		normal = diff1.cross(diff2);

	} catch (Exception e){
		normal = Normal(0.0, 0.0, 0.0);
		//cout << "Exception: " << e.what() << endl;
	}

	//cout << "Plane" << endl;

	Plane p;
	p.a = C(1);
	p.b = C(2);
	p.c = C(3);
	p.n = normal;
	p.p = query_point;

	return p;
}


void StannInterpolator::write_normals(){

	ofstream out("normals.nor");
	for(int i = 0; i < number_of_points; i++){
		if(i % 10000 == 0) cout << "##### Writing points and normals: " << i
		                       << " / " << number_of_points << endl;
		out << points[i][0] << " " << points[i][1] << " " << points[i][2] << " "
		    << normals[i][0] << " " << normals[i][1] << " " << normals[i][2] << endl;
	}

}

void StannInterpolator::estimate_normals(){


	int k_0 = 10;
	//int k_0 = 50;

	cout << "##### Initializing normal array..." << endl;

	//Initialize normal array
	normals = new float*[number_of_points];

	cout << "##### Estimating normals for all points..." << endl << endl;
	boost::progress_display progress(number_of_points);

    #pragma omp parallel for
	for(int i = 0; i < number_of_points; i++){

		Vertex query_point; //, diff1, diff2;
		Normal normal;

		vector<unsigned long> id;
		vector<double> di;
		float mean_distance;

		int n = 0;
		int k = k_0;

		while(n < 5){

			n++;
			k = k * 2;

			point_tree.ksearch(points[i], k, id, di, 0);

			float min_x = 1e15;
			float min_y = 1e15;
			float min_z = 1e15;
			float max_x = - min_x;
			float max_y = - min_y;
			float max_z = - min_z;

			float dx, dy, dz;
			dx = dy = dz = 0;


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
			//break;

 		}

		query_point = Vertex(points[i][0], points[i][1], points[i][2]);
		Plane p = calcPlane(query_point, k, id);

		mean_distance = meanDistance(p, id, k);

		normal =  p.n;
		if(normal * (query_point - center) < 0) normal = normal * -1;

		normals[i] = new float[3];
		normals[i][0] = normal[0];
		normals[i][1] = normal[1];
		normals[i][2] = normal[2];

		++progress;
	}


	cout << endl << endl;

	//interpolateNormals(20);
}

void StannInterpolator::interpolateNormals(int k){

	vector<Normal> tmp;

	for(int i = 0; i < number_of_points; i++) tmp.push_back(Normal());

	cout << "##### Interpolating normals..." << endl << endl;

	boost::progress_display progress(number_of_points);

	#pragma omp parallel for
	for(int i = 0; i < number_of_points; i++){

		vector<unsigned long> id;
		vector<double> di;

		point_tree.ksearch(points[i], k, id, di, 0);

		//if(i % 10000 == 0) cout << "##### Interpolating normals: "
		//<< i << " / " << number_of_points << endl;

//		if(i % progress_limit == 0)
//		{
//			cout << "##### Interpolating normals: " << counter << "% " << endl;
//			counter++;
//		}

		Vertex mean;
		Normal mean_normal;

		for(int j = 0; j < k; j++){
			mean += Vertex(normals[id[j]][0],
					normals[id[j]][1],
					normals[id[j]][2]);
		}
		mean_normal = Normal(mean);

		tmp[i] = mean;

		for(int j = 0; j < k; j++){
			Normal n = Normal(normals[id[j]][0], normals[id[j]][1], normals[id[j]][2]);
			if(fabs(n * mean_normal) > 0.2 ){
				normals[id[j]][0] = mean_normal[0];
				normals[id[j]][1] = mean_normal[1];
				normals[id[j]][2] = mean_normal[2];
			}
		}
		++progress;
	}
	cout << endl << endl;

	cout << "##### Copying normals..." << endl;

	for(int i = 0; i < number_of_points; i++){
		normals[i][0] = tmp[i][0];
		normals[i][1] = tmp[i][1];
		normals[i][2] = tmp[i][2];
	}

}


float StannInterpolator::distance(ColorVertex v){

	int k = 40;


	vector<unsigned long> id;
	vector<double> di;

	//Allocate ANN point
	float * p;
	p = new float[3];
	p[0] = v.x; p[1] = v.y; p[2] = v.z;

	//Find nearest tangent plane
	point_tree.ksearch(p, k, id, di, 0);

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

	delete[] p;

	return distance;
}
