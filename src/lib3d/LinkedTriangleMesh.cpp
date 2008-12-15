/*
 * LinkedTriangleMesh.cpp
 *
 *  Created on: 15.12.2008
 *      Author: twiemann
 */

#include "LinkedTriangleMesh.h"

LinkedTriangleMesh::LinkedTriangleMesh() {
	// TODO Auto-generated constructor stub

}

LinkedTriangleMesh::~LinkedTriangleMesh() {
	// TODO Auto-generated destructor stub
}

void LinkedTriangleMesh::finalize(){

	number_of_vertices = (int)vertex_buffer.size();
	number_of_faces    = (int)triangle_buffer.size();

	normals  = new float[3 * number_of_vertices];
	vertices = new float[3 * number_of_vertices];
	colors   = new float[3 * number_of_vertices];

	indices  = new unsigned int[3 * number_of_faces];

	for(int i = 0; i < number_of_vertices; i++){
		for(int j = 0; j < 3; j++){
			normals [3 * i + j] = -normal_buffer[i][j];
			vertices[3 * i + j] = vertex_buffer[i][j];
		}
		colors[3 * i    ] = 0.0f;
		colors[3 * i + 1] = 1.0f;
		colors[3 * i + 2] = 0.0f;
	}

	int iii;

	for(size_t i = 0; i < triangle_buffer.size(); i++){
		iii = 3 * i;
		indices[iii    ] = triangle_buffer[i].getIndex(0);
		indices[iii + 1] = triangle_buffer[i].getIndex(1);
		indices[iii + 2] = triangle_buffer[i].getIndex(2);

	}

	vertex_buffer.clear();
	normal_buffer.clear();
	triangle_buffer.clear();

	finalized = true;

}

void LinkedTriangleMesh::addTriangle(int v0, int v1, int v2){
	LinkedTriangle t(this, v0, v1, v2);
	triangle_buffer.push_back(t);
}
