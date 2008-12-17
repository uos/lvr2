/*
 * TriangleMesh.cpp
 *
 *  Created on: 17.10.2008
 *      Author: Thomas Wiemann
 */

#include "TriangleMesh.h"

TriangleMesh::TriangleMesh(): StaticMesh() {

}

TriangleMesh::~TriangleMesh(){

	if(colors   != 0) delete[] colors;
	if(vertices != 0) delete[] vertices;
	if(indices  != 0) delete[] indices;
	if(normals  != 0) delete[] normals;

	index_buffer.clear();
	vertex_buffer.clear();
	normal_buffer.clear();

}

void TriangleMesh::finalize(){

	number_of_vertices = (int)vertex_buffer.size();
	number_of_faces    = (int)index_buffer.size() / 3;

	normals =  new float[3 * number_of_vertices];
	vertices = new float[3 * number_of_vertices];
	indices =  new unsigned int[number_of_faces * 3];
	colors = new float [3 * number_of_vertices];

	for(int i = 0; i < number_of_vertices; i++){
		for(int j = 0; j < 3; j++){
			normals [3 * i + j] = -normal_buffer[i][j];
			vertices[3 * i + j] = vertex_buffer[i][j];
		}
		colors[3 * i    ] = 0.0f;
		colors[3 * i + 1] = 1.0f;
		colors[3 * i + 2] = 0.0f;
	}

	for(size_t i = 0; i < index_buffer.size(); i++){
		indices[i] = index_buffer[i];
	}

	vertex_buffer.clear();
	normal_buffer.clear();
	index_buffer.clear();

	finalized = true;

}

Vertex TriangleMesh::getVertex(int n){
	assert(n < vertex_buffer.size());
	return vertex_buffer[n];
}

void TriangleMesh::printStats(){

	if(finalized){
		cout << "##### Triangle Mesh (S): " << number_of_vertices << " Vertices / "
		                                    << number_of_faces    << " Faces.   " << endl;
	} else {
		cout << "##### Triangle Mesh (D): " << vertex_buffer.size() << " Vertices / "
		                                    << index_buffer.size() / 3 << " Faces." << endl;
	}

}

void TriangleMesh::interpolateNormal(Normal n, size_t index){

	if(index < normal_buffer.size()){

		Normal normal = normal_buffer[index];
		normal += n;
		normal.normalize();

		normal_buffer[index] = normal;

	} else {
		cout << "WARNING: TriangleMesh: Normal index out of range: " << index << endl;
	}

}
