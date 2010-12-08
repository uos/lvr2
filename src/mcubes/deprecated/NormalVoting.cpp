/*
 * NormalVoting.cpp
 *
 *  Created on: 05.03.2009
 *      Author: twiemann
 */

#include "NormalVoting.h"

NormalVoting::NormalVoting(string filename, float vs) {

	readPoints(filename);
	voxelsize = vs;

	interpolator = new StannInterpolator(points, number_of_points, 10, 100, 100.0);

	vote();
	save("biggest.nor");

}

void NormalVoting::save(string filename){

	ofstream out(filename.c_str());

	if(out.good()){

		for(int i = 0; i < buckets[biggest_bucket].normals.size(); i++){
			Vertex position = buckets[biggest_bucket].vertices[i];
			Normal normal   = buckets[biggest_bucket].normals[i];
			out << position.x << " " << position.y << " " << position.z << " ";
			out << normal.x << " " << normal.y << " " << normal.z << endl;
		}

	}

}

void NormalVoting::vote(){

	biggest_bucket = 0;

	int biggest_bucket_size = 0;

	Normal n;
	Vertex v;


	//Initialize first bucket
	n = interpolator->normals[0];
	v = Vertex(interpolator->points[0][0],
			interpolator->points[0][1],
			interpolator->points[0][2]);

	buckets.push_back(NormalBucket(n, v));

	//Iterate through normals
	bool found;
	for(int i = 1; i < number_of_points; i++){

		found = false;

		n = interpolator->normals[i];
		v = Vertex(	interpolator->points[i][0],
					interpolator->points[i][1],
					interpolator->points[i][2]);

		//Check all buckets
		for(int j = 0; j < buckets.size(); j++){
			if(buckets[j].insert(n, v)){
				found = true;
				break;
			}
		}

		//If normal was not insertet, crate new bucket
		if(!found){
			buckets.push_back(NormalBucket(n, v));
		}

		if(i % 1000 == 0) cout << "Voting: " << i << " / " << number_of_points << " Buckets: " << buckets.size() << endl;
	}

	//DEBUG: PRINT BUCKET SIZES:

	for(int i = 0; i < buckets.size(); i++){
		if(buckets[i].normals.size() > biggest_bucket_size){
			biggest_bucket_size = buckets[i].normals.size();
			biggest_bucket = i;
		}
		cout << i << endl;
	}

}

void NormalVoting::readPoints(string filename){

	ifstream in(filename.c_str());

	//Vector to tmp-store points in file
	vector<BaseVertex> pts;

	//Read all points. Save maximum and minimum dimensions and
	//calculate maximum indices.
	int c = 0;

	//Get number of data fields to ignore
	int number_of_dummys = getFieldsPerLine(filename) - 3;

	//Point coordinates
	float x, y, z, dummy;

	//Read file
	while(in.good()){
		in >> x >> y >> z;
		for(int i = 0; i < number_of_dummys; i++){
			in >> dummy;
		}


		//bounding_box.expand(x, y, z);
		pts.push_back(BaseVertex(x,y,z));
		c++;

		if(c % 10000 == 0) cout << "##### Reading Points... " << c << endl;
	}

	cout << "##### Finished Reading. Number of Data Points: " << pts.size() << endl;


	//Create ANNPointArray
	cout << "##### Creating ANN Points " << endl;
	points = annAllocPts(c, 3);

	for(size_t i = 0; i < pts.size(); i++){
		points[i][0] = pts[i].x;
		points[i][1] = pts[i].y;
		points[i][2] = pts[i].z;
	}

	pts.clear();
	number_of_points = c;

}

int NormalVoting::getFieldsPerLine(string filename){

	ifstream in(filename.c_str());

	//Get first line from file
	char first_line[1024];
	in.getline(first_line, 1024);
	in.close();

	//Get number of blanks
	int c = 0;
	char* pch = strtok(first_line, " ");
	while(pch != NULL){
		c++;
		pch = strtok(NULL, " ");
	}

	in.close();

	return c;
}

NormalVoting::~NormalVoting() {
	delete interpolator;
}
