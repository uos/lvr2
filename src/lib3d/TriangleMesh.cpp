/*
 * TriangleMesh.cpp
 *
 *  Created on: 13.10.2008
 *      Author: twiemann
 */

#include "TriangleMesh.h"

TriangleMesh::TriangleMesh(string filename) {

	listIndex = -1;

	vertices  = 0;
	normals   = 0;
	colors    = 0;
	indices   = 0;

	number_of_vertices = 0;
	number_of_faces    = 0;

	readPLY(filename);

	initDisplayList();
}

void TriangleMesh::initDisplayList(){

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);

	listIndex = glGenLists(1);

	glNewList(listIndex, GL_COMPILE);
	glColorPointer(3, GL_FLOAT, 0, colors);
	glNormalPointer(GL_FLOAT, 0, normals);
	glVertexPointer(3, GL_FLOAT, 0, vertices);

	//glDrawElements(GL_TRIANGLES, 3 * number_of_faces, GL_UNSIGNED_INT, indices);
	glDrawArrays(GL_TRIANGLES, 0, number_of_vertices);
	glEndList();
}

TriangleMesh::~TriangleMesh() {

	if(vertices != 0) delete[] vertices;
	if(normals  != 0) delete[] normals;
	if(colors   != 0) delete[] colors;
	if(indices  != 0) delete[] indices;

}

void TriangleMesh::readPLY(string filename){

	ifstream in;

	PlyHeaderDescription head;
	PlyVertexDescription vertex_dcr;
	PlyFaceDescription face_dcr;

	PlyFace ply_face;
	PlyVertex ply_vertex;

	in.open(filename.c_str(), fstream::in | fstream::binary);

	if(!in.good()){
		cout << "!!!!! Warning: PLY-Reader: Cannot open file' " << filename << "'." << endl;
		return;
	}

	cout << "##### PLY-Reader: Reading " << filename << "..." << endl;

	in.read( (char*)&head, sizeof(head));
	in.read( (char*)&vertex_dcr, sizeof(vertex_dcr));
	in.read( (char*)&face_dcr, sizeof(face_dcr));

	const char* buffer = "end_header\n";
	char dummy[20];
	in.read( dummy, strlen(buffer));

	vertices = new float[3 * vertex_dcr.count];
	normals  = new float[3 * vertex_dcr.count];
	colors   = new float[3 * vertex_dcr.count];

	int index;

	number_of_vertices = vertex_dcr.count;

	for(unsigned int i = 0; i < vertex_dcr.count; i++){

		index = 3 * i;

		in.read( (char*)&ply_vertex, sizeof(PlyVertex));

		vertices[index    ] = ply_vertex.x;
		vertices[index + 1] = ply_vertex.y;
		vertices[index + 2] = ply_vertex.z;

		colors  [index    ] = ply_vertex.r;
		colors  [index + 1] = ply_vertex.g;
		colors  [index + 2] = ply_vertex.b;

		normals [index    ] = ply_vertex.nx;
		normals [index + 1] = ply_vertex.ny;
		normals [index + 2] = ply_vertex.nz;

	}

	number_of_faces = face_dcr.count;

	indices = new unsigned int[3 * number_of_faces];

	for(unsigned int i = 0; i < face_dcr.count; i++){

		index = 3 * i;

		in.read( (char*)&ply_face, sizeof(ply_face));
		for(int j = 0; j < 3; j++) indices[index + j] = ply_face.indices[j];
	}

}
