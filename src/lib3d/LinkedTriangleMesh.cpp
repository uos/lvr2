/*
 * LinkedTriangleMesh.cpp
 *
 *  Created on: 15.12.2008
 *      Author: twiemann
 */



#include "LinkedTriangleMesh.h"

LinkedTriangleMesh::LinkedTriangleMesh() : TriangleMesh(){
	num_verts = num_triangles = 0;
	vertex_buffer.clear();
	triangle_buffer.clear();
}

LinkedTriangleMesh::LinkedTriangleMesh(const LinkedTriangleMesh& m)
{
	num_verts = m.num_verts;
	num_triangles = m.num_triangles;

	vertex_buffer = m.vertex_buffer;        // NOTE: triangles are still pointing to original mesh
	triangle_buffer = m.triangle_buffer;
                                            // NOTE: should reset triangles in vertex_buffer and
	                                        //       triangle buffer
}

LinkedTriangleMesh::~LinkedTriangleMesh() {
	num_verts = num_triangles = 0;
	vertex_buffer.clear();
	triangle_buffer.clear();
}

void LinkedTriangleMesh::addVertex(Vertex v){
	static int i = 0;

	TriangleVertex vert(v);
	vert.setIndex(i);

	vertex_buffer.push_back(vert);
	i++;
	num_verts = i;
}

void LinkedTriangleMesh::addTriangle(int v1, int v2, int v3){
	static int i = 0;
	assert(v1 < num_verts && v2 < num_verts && v3 < num_verts);

	LinkedTriangle t(this, v1, v2 , v3);
	t.setFaceIndex(i);
	t.calculateNormal();

	triangle_buffer.push_back(t);

	vertex_buffer[v1].addTriangleNeighbor(i);
	vertex_buffer[v1].addVertexNeighbor(v2);
	vertex_buffer[v1].addVertexNeighbor(v3);

	vertex_buffer[v2].addTriangleNeighbor(i);
	vertex_buffer[v2].addVertexNeighbor(v1);
	vertex_buffer[v2].addVertexNeighbor(v3);

	vertex_buffer[v3].addTriangleNeighbor(i);
	vertex_buffer[v3].addVertexNeighbor(v1);
	vertex_buffer[v3].addVertexNeighbor(v2);

	num_triangles = i;
	i++;
}



void LinkedTriangleMesh::calcBoundingBox(BoundingBox& b){
	for(size_t i = 0; i < vertex_buffer.size(); i++){
		b.expand(vertex_buffer[i].getPosition());
	}
}

void LinkedTriangleMesh::normalize(){

	BoundingBox b;
	calcBoundingBox(b);

	float scale;

	Vertex vmin = b.v_min;
	Vertex vmax = b.v_max;
	Vertex diff = vmax - vmin;

	if      (diff.x >= diff.y && diff.x >= diff.z) scale = 2.0f / diff.x;
	else if (diff.y >= diff.x && diff.y >= diff.z) scale = 2.0f / diff.y;
	else    scale = 2.0f / diff.z;

	Vertex translation = (vmin + vmax) * 0.5f;

	for(size_t i = 0; i < vertex_buffer.size(); i++){
		vertex_buffer[i].position -= translation;
		vertex_buffer[i].position *= scale;
	}

}

void LinkedTriangleMesh::finalize(){

	int number_of_active_triangles = 0;

	if(finalized){
		cout << "Warning: StaticMesh::finalize(): Mesh is already finalized." << endl;
	}

	for(int i = 0; i < (int)triangle_buffer.size(); i++){
		if(triangle_buffer[i].isActive()) number_of_active_triangles++;
	}

	cout << "NUMBER OF ACTIVE TRIANGLES: " << number_of_active_triangles << endl << flush;

	number_of_vertices = (int)vertex_buffer.size();
	number_of_faces    = number_of_active_triangles;

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

	size_t iii;

	for(size_t i = 0; i < triangle_buffer.size(); i++){
		if(triangle_buffer[i].isActive()){
			iii = 3 * i;
			indices[iii    ] = triangle_buffer[i].getIndex(0);
			indices[iii + 1] = triangle_buffer[i].getIndex(1);
			indices[iii + 2] = triangle_buffer[i].getIndex(2);
		}
	}

//	vertex_buffer.clear();
//	normal_buffer.clear();
//	triangle_buffer.clear();

	finalized = true;

}



