/*
 * StaticMesh.cpp
 *
 *  Created on: 12.11.2008
 *      Author: twiemann
 */

#include "StaticMesh.h"

StaticMesh::StaticMesh(){

	normals = 0;
	vertices = 0;
	colors = 0;
	indices = 0;

	number_of_faces = 0;
	number_of_vertices = 0;

	finalized = false;

}

StaticMesh::StaticMesh(string name) : Renderable(name){

	normals = 0;
	vertices = 0;
	colors = 0;
	indices = 0;

	number_of_faces = 0;
	number_of_vertices = 0;

	finalized = false;

	load(name);

}

StaticMesh::StaticMesh(const StaticMesh &o){

	if(normals != 0) delete[] normals;
	if(vertices != 0) delete[] vertices;
	if(colors != 0) delete[] colors;
	if(indices != 0) delete[] indices;

	normals = new float[3 * o.number_of_vertices];
	vertices = new float[3 * o.number_of_vertices];
	colors = new float[3 * o.number_of_vertices];

	indices = new unsigned int[3 * number_of_faces];

	for(int i = 0; i < 3 * number_of_vertices; i++){
		normals[i] = o.normals[i];
		vertices[i] = o.vertices[i];
		colors[i] = o.colors[i];
	}

	for(int i = 0; i < 3 * number_of_faces; i++){
		indices[i] = o.indices[i];
	}

}

void StaticMesh::compileDisplayList(){

	if(finalized){

		listIndex = glGenLists(1);

		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_NORMAL_ARRAY);
		glEnableClientState(GL_COLOR_ARRAY);

		glNewList(listIndex, GL_COMPILE);
		glVertexPointer(3, GL_FLOAT, 0, vertices);
		glNormalPointer(GL_FLOAT, 0, normals);
		glColorPointer(3, GL_FLOAT, 0, colors);
		glDrawElements(GL_TRIANGLES, number_of_faces, GL_UNSIGNED_INT, indices);
		glEndList();

	}

}

void StaticMesh::load(string filename){

}

void StaticMesh::save(string filename){

}
