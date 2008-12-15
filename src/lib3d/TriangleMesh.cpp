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

	triangle_buffer.clear();
	vertex_buffer.clear();
	normal_buffer.clear();
}

void TriangleMesh::addTriangle(int v0, int v1, int v2){
	Triangle t(this, v0, v1, v2);
	triangle_buffer.push_back(t);
}

void TriangleMesh::finalize(){

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

Vertex TriangleMesh::getVertex(int index){
	assert(index < (int)vertex_buffer.size());
	return vertex_buffer[index];
}

Normal TriangleMesh::getNormal(int index){
	assert(index < (int)normal_buffer.size());
	return normal_buffer[index];
}

void TriangleMesh::printStats(){

	if(finalized){
		cout << "##### Triangle Mesh (S): " << number_of_vertices << " Vertices / "
		                                    << number_of_faces    << " Faces.   " << endl;
	} else {
		cout << "##### Triangle Mesh (D): " << vertex_buffer.size() << " Vertices / "
		                                    << triangle_buffer.size() / 3 << " Faces." << endl;
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

TriangleMesh& TriangleMesh::operator=(const TriangleMesh& m){
	if(this == &m) return *this;

	//Assign buffers
	vertex_buffer      = m.vertex_buffer;
	normal_buffer      = m.normal_buffer;
	triangle_buffer    = m.triangle_buffer;

	//Assign buffer lengths
	number_of_vertices = m.number_of_vertices;
	number_of_faces    = m.number_of_faces;

	//Assign static buffers
	normals            = m.normals;
	vertices           = m.vertices;
	colors             = m.colors;
	indices            = m.indices;
	finalized          = m.finalized;

	//Assign Renderable stuff
	visible            = m.visible;
    transformation     = m.transformation;
	name               = m.name;
	listIndex          = m.listIndex;
	axesListIndex      = m.axesListIndex;

	x_axis             = m.x_axis;
	y_axis             = m.y_axis;
	z_axis             = m.z_axis;

	position           = m.position;
	rotation_speed     = m.rotation_speed;
	translation_speed  = m.translation_speed;

    show_axes          = m.show_axes;
	active             = m.active;
    scale_factor       = m.scale_factor;

    return *this;
}
