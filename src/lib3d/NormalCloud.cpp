/*
 * NormalCloud.cpp
 *
 *  Created on: 13.10.2008
 *      Author: Thomas Wiemann
 */

#include "NormalCloud.h"

NormalCloud::NormalCloud(string filename) {

	ifstream in(filename.c_str());

	if(!in.good()){
		cout << "##### NormalCloud: Error: Could not open file " << filename << "." << endl;
		return;
	}

	int i = 0;
	float x, y, z, nx, ny, nz;
	while(in.good()){
		if(i > 0 && i % 100000 == 0) cout << "##### Normal Cloud: READING POINTS: " << i << endl;
		in >> x >> y >> z >> nx >> ny >> nz;
		points.push_back(Vertex(x,y,z));
		normals.push_back(Normal(nx, ny, nz));
		i++;
	}

	initDisplayList();

}

NormalCloud::NormalCloud(const NormalCloud &o){

	points.clear();
	normals.clear();

	points = o.points;
	normals = o.normals;

}

NormalCloud::~NormalCloud() {
	// TODO Auto-generated destructor stub
}

void NormalCloud::initDisplayList(){

	listIndex = glGenLists(1);
	glNewList(listIndex, GL_COMPILE);
	glDisable(GL_LIGHTING);
	glPointSize(4.0);
	glBegin(GL_POINTS);
	glColor3f(0.0, 1.0, 0.0);
	for(size_t i = 0; i < points.size(); i++){
		glVertex3f(points[i].x,
				   points[i].y,
				   points[i].z);
	}
	glEnd();
	glPointSize(2.0);

	glLineWidth(1.5);
	for(size_t i = 0; i < normals.size(); i++){
		glBegin(GL_LINES);
		glColor3f(1.0, 0.0, 0.0);
		glVertex3f(points[i][0],
				   points[i][1],
				   points[i][2]);
		glVertex3f(points[i][0] + 3 * normals[i][0],
				   points[i][1] + 3 * normals[i][1],
				   points[i][2] + 3 * normals[i][2]);
		glEnd();
	}
	glLineWidth(1.0);
	glEnable(GL_LIGHTING);
	glEndList();

}
