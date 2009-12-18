/*
 * PointCloud.cpp
 *
 *  Created on: 02.09.2008
 *      Author: twiemann
 */

#include "PointCloud.h"

PointCloud::PointCloud(string filename) : Renderable(filename){

	ifstream in(filename.c_str());

	if(!in.good()){
		cout << "##### Error: Could not open file " << filename << "." << endl;
		return;
	}

	int i = 0;
	float x, y, z, dummy;
	while(in.good()){
		if(i > 0 && i % 100000 == 0) cout << "##### READING POINTS: " << i << endl;
		in >> x >> y >> z >> dummy;
		points.push_back(Vertex(x,y,z));
		i++;
	}
	initDisplayList();
}

void PointCloud::initDisplayList(){

	listIndex = glGenLists(1);
	glNewList(listIndex, GL_COMPILE);
	glBegin(GL_POINTS);
	for(size_t i = 0; i < points.size(); i++){
		glVertex3f(points[i].x,
				   points[i].y,
				   points[i].z);
	}
	glEnd();
	glEndList();
}


PointCloud::~PointCloud() {

}
