/*
 * PolygonMesh.cpp
 *
 *  Created on: 03.02.2009
 *      Author: twiemann
 */

#include "PolygonMesh.h"

PolygonMesh::PolygonMesh(string filename) : Renderable(filename){
	load(filename);
	compileDisplayList();
}

void PolygonMesh::load(string filename){

	char line[1024];
	Lib3D::Polygon p;

	ifstream in(filename.c_str());

	if(!in.good()){
		cout << "Warning: PolygonMesh::load(): Could not open file '"
		     << filename << "'." << endl;
		return;
	}

	float x,y,z;
	float r,g,b;

	int c = 0;
	while(in.good()){
		in.getline(line, 1024);
		if(strstr(line, "BEGIN") != NULL){
			if(c % 1000 == 0){
				cout << "PolygonMesh: Read " << c << " polygons." << endl;
			}
			p = Lib3D::Polygon();
			c++;

		} else if(strstr(line, "END") != NULL) {
			polygons.push_back(p);
		} else {
			x = y = z = 0.0;
			r = g = b = 0.0;

			sscanf(line, "%f %f %f %f %f %f", &x, &y, &z, &r, &g, &b);
			p.addVertex(Vertex(x, y, z));
		}
	}

}

void PolygonMesh::save(string filename){

	cout << "TO DO: Implement PolygonMesh::save()!!!" << endl;

}

void PolygonMesh::compileDisplayList(){

	Lib3D::Polygon p;

	listIndex = glGenLists(1);
	glNewList(axesListIndex, GL_COMPILE);

	glDisable(GL_LIGHTING);
	glColor3f(0.0, 0.0, 0.0);
	for(size_t i = 0; i < polygons.size(); i++){
		p = polygons[i];
		glBegin(GL_LINES);
		for(size_t j = 0; j < p.vertices.size(); j++){
			glVertex3f(p.vertices[j].x, p.vertices[j].y, p.vertices[j].z);
		}
		glEnd();
	}
	glEnable(GL_LIGHTING);
	glEndList();

}

PolygonMesh::~PolygonMesh() {
	// TODO Auto-generated destructor stub
}
