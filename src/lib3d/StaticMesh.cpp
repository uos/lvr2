/*
 * StaticMesh.cpp
 *
 *  Created on: 12.11.2008
 *      Author: twiemann
 */

#include "StaticMesh.h"
#include "PLYWriter.h"

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

	indices = new unsigned int[3 * o.number_of_faces];

	for(int i = 0; i < 3 * o.number_of_vertices; i++){
		normals[i] = o.normals[i];
		vertices[i] = o.vertices[i];
		colors[i] = o.colors[i];
	}

	for(int i = 0; i < 3 * o.number_of_faces; i++){
		indices[i] = o.indices[i];
	}

}

StaticMesh::~StaticMesh(){

}

void StaticMesh::finalize(){

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
		glDrawElements(GL_TRIANGLES, 3 * number_of_faces, GL_UNSIGNED_INT, indices);
		glEndList();

	}

}

void StaticMesh::load(string filename){

	ifstream in;

	PlyHeaderDescription head;
	PlyVertexDescription vertex_dcr;
	PlyFaceDescription face_dcr;

	PlyFace ply_face;
	PlyVertex ply_vertex;

	in.open(filename.c_str(), fstream::in | fstream::binary);

	in.read( (char*)&head, sizeof(head));
	in.read( (char*)&vertex_dcr, sizeof(vertex_dcr));
	in.read( (char*)&face_dcr, sizeof(face_dcr));

	char* buffer = "end_header\n";
	char dummy[20];
	in.read( dummy, (streamsize)strlen(buffer));

	//Save no. of vertices and faces
	number_of_vertices = vertex_dcr.count;
	number_of_faces = face_dcr.count;

	//Create Arrays
	if(normals != 0) delete[] normals;
	if(vertices != 0) delete[] vertices;
	if(colors != 0) delete[] colors;
	if(indices != 0) delete[] indices;

	normals = new float[3 * number_of_vertices];
	vertices = new float[3 * number_of_vertices];
	colors = new float[3 * number_of_vertices];

	indices = new unsigned int[3 * number_of_faces];

	for(unsigned int i = 0; i < vertex_dcr.count; i++){

		in.read( (char*)&ply_vertex, sizeof(PlyVertex));

		vertices[3 * i    ] = ply_vertex.x;
		vertices[3 * i + 1] = ply_vertex.y;
		vertices[3 * i + 2] = ply_vertex.z;

		normals [3 * i    ] = ply_vertex.nx;
		normals [3 * i + 1] = ply_vertex.ny;
		normals [3 * 1 + 2] = ply_vertex.nz;

		colors  [3 * i    ] = ply_vertex.r;
		colors  [3 * i + 1] = ply_vertex.g;
		colors  [3 * i + 2] = ply_vertex.b;

	}

	for(unsigned int i = 0; i < face_dcr.count; i++){
		in.read( (char*)&ply_face, sizeof(ply_face));
		for(int j = 0; j < 3; j++){
			indices[3 * i + j] = ply_face.indices[j];
			if(indices[3 * i + j] >= (unsigned int)number_of_vertices ||indices[3 * i + j] < 0){
				cout << indices[3 * i + j] << " " << number_of_vertices << endl;
				cout << "ERROR!" << endl << flush;
			}
		}

	}

	cout << "LOAD COMPLETE" << endl << flush;

	finalized = true;
	compileDisplayList();

}

void StaticMesh::save(string filename){

	if(finalized){

		cout << 1 << endl << flush;
		ofstream out;
		out.open(filename.c_str(), fstream::out | fstream::binary);

		if(!out.good()){


			cout << "Static Mesh::Save: Unable to open file: " << filename << endl;
			return;
		}

		char* comment = "comment c\n";


		//*********** WRITE PLY HEADER ***************

		PlyHeaderDescription header_dcr;
		PlyVertexDescription vertex_dcr;
		PlyFaceDescription face_dcr;

		//Setting up header
		strcpy(header_dcr.ply, "ply");
		strcpy(header_dcr.format, PLY_LITTLE_ENDIAN);
		strcpy(header_dcr.comment, comment);

		//Setting up vertex description
		strcpy(vertex_dcr.element, "element vertex ");

		strcpy(vertex_dcr.property_x, "property float x\n");
		strcpy(vertex_dcr.property_y, "property float y\n");
		strcpy(vertex_dcr.property_z, "property float z\n");

		strcpy(vertex_dcr.property_nx, "property float nx\n");
		strcpy(vertex_dcr.property_ny, "property float ny\n");
		strcpy(vertex_dcr.property_nz, "property float nz\n");
		vertex_dcr.count = 0;


		//Setting up face description
		strcpy(face_dcr.face, "face ");
		strcpy(face_dcr.property, "property list uchar int vertex_index\n");
		face_dcr.count = 3;

		//Local variables
		PlyVertex ply_vertex;
		PlyFace ply_face;

		cout << "NUMBER OF FACES: " << number_of_faces << endl << flush;

		//Set vertex and face count
		vertex_dcr.count = number_of_vertices;
		face_dcr.count = number_of_faces;
		ply_face.vertexCount = 0;



		//Write header
		out.write( (char*)&header_dcr, sizeof(header_dcr));
		out.write( (char*)&vertex_dcr, sizeof(vertex_dcr));
		out.write( (char*)&face_dcr, sizeof(face_dcr));

		char* buffer = "end_header\n";
		out.write(buffer, (streamsize)strlen(buffer));

		//Write vertices and normals
		for(unsigned int i = 0; i < vertex_dcr.count; i++){

			ply_vertex.x = vertices[3 * i];
			ply_vertex.y = vertices[3 * i + 1];
			ply_vertex.z = vertices[3 * i + 2];

			ply_vertex.nx = normals[3 * i];
			ply_vertex.ny = normals[3 * i + 1];
			ply_vertex.nz = normals[3 * 1 + 2];

			ply_vertex.r = colors[3 * i];
			ply_vertex.g = colors[3 * i + 1];
			ply_vertex.b = colors[3 * i + 2];

			ply_vertex.u = 0.0;
			ply_vertex.v = 0.0;

			ply_vertex.texture = 1;

			out.write( (char*)&ply_vertex, sizeof(ply_vertex));
		}

		for(int i = 0; i < number_of_faces; i ++){

			ply_face.indices[0] = indices[3 * i];
			ply_face.indices[1] = indices[3 * i + 1];
			ply_face.indices[2] = indices[3 * i + 2];

			out.write( (char*)&ply_face, sizeof(ply_face));
		}
	} else {
		cout << "##### Warning: Static Mesh: Buffers empty." << endl;
	}

}

void StaticMesh::savePLY(string filename)
{
//	// Test if mesh is finalized
//	if(finalized)
//	{
//		PLYWriter writer(filename);
//
//		// Generate vertex element description in .ply header section
//
//		// Add properties depending on the available buffers
//		if(vertices != 0)
//		{
//
//		}
//
//		if(normals != 0)
//		{
//
//		}
//
//	}
//	else
//	{
//		cout << "#### Warning: Static Mesh: Buffers empty." << endl;
// 	}

}
