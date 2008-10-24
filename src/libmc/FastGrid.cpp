/*
 * FastGrid.cpp
 *
 *  Created on: 22.10.2008
 *      Author: Thomas Wiemann
 */

#include "FastGrid.h"

FastGrid::FastGrid(string filename, float vs) {

	voxelsize = vs;
	number_of_points = 0;

	readPoints(filename);
	calcIndices();
}

FastGrid::~FastGrid() {
	annDeallocPts(points);
}


void FastGrid::createGrid(){

	//Create Grid
	cout << "##### Creating Grid..." << endl;

	//Current indices
	int index_x, index_y, index_z;
	int dx, dy, dz;
	int hash_value;


	//Iterators
	hash_map<int, FastBox*>::iterator it;
	hash_map<int, FastBox*>::iterator neighbour_it;

	for(int i = 0; i < number_of_points; i++){

		index_x = calcIndex((points[i][0] - bounding_box.v_min.x) / voxelsize);
		index_y = calcIndex((points[i][1] - bounding_box.v_min.y) / voxelsize);
		index_z = calcIndex((points[i][2] - bounding_box.v_min.z) / voxelsize);



	}

	cout << "##### Finished Grid Creation. Number of generated cells: " << cells.size() << endl;

}

void FastGrid::createMesh(){

}

void FastGrid::calcIndices(){

	float max_size = max(max(bounding_box.x_size, bounding_box.y_size), bounding_box.z_size);

	//Save needed grid parameters
	max_index = (int)ceil( (max_size + 5 * voxelsize) / voxelsize);
	max_index_square = max_index * max_index;

	max_index_x = (int)ceil(bounding_box.x_size / voxelsize) + 1;
	max_index_y = (int)ceil(bounding_box.y_size / voxelsize) + 2;
	max_index_z = (int)ceil(bounding_box.z_size / voxelsize) + 3;

}


void FastGrid::readPoints(string filename){

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

		bounding_box.expand(x, y, z);
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

int FastGrid::getFieldsPerLine(string filename){

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

