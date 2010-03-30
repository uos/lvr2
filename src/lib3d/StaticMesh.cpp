/*
 * StaticMesh.cpp
 *
 *  Created on: 12.11.2008
 *      Author: Thomas Wiemann
 */

#include "StaticMesh.h"
#include "PLYWriter.h"

#include <cassert>

StaticMesh::StaticMesh(){

	normals = 0;
	vertices = 0;
	colors = 0;
	m_indices = 0;

	number_of_faces = 0;
	number_of_vertices = 0;

	finalized = false;

}

StaticMesh::StaticMesh(string name) : Renderable(name){

	normals = 0;
	vertices = 0;
	colors = 0;
	m_indices = 0;

	number_of_faces = 0;
	number_of_vertices = 0;

	finalized = false;

	load(name);

}

StaticMesh::StaticMesh(const StaticMesh &o){

	if(normals != 0) delete[] normals;
	if(vertices != 0) delete[] vertices;
	if(colors != 0) delete[] colors;
	if(m_indices != 0) delete[] m_indices;

	normals = new float[3 * o.number_of_vertices];
	vertices = new float[3 * o.number_of_vertices];
	colors = new float[3 * o.number_of_vertices];

	m_indices = new unsigned int[3 * o.number_of_faces];



	for(size_t i = 0; i < 3 * o.number_of_vertices; i++){
		normals[i] = o.normals[i];
		vertices[i] = o.vertices[i];
		colors[i] = o.colors[i];
	}

	for(size_t i = 0; i < 3 * o.number_of_faces; i++){
		m_indices[i] = o.m_indices[i];
	}

}

StaticMesh::~StaticMesh(){

}

void StaticMesh::finalize(){

}

void StaticMesh::compileDisplayList(){

	if(finalized){

		listIndex = glGenLists(1);

		// Enable vertex / normal / color arrays
		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_NORMAL_ARRAY);
		glEnableClientState(GL_COLOR_ARRAY);

		// Start new display list
		glNewList(listIndex, GL_COMPILE);

		// Assign element pointers
		glVertexPointer(3, GL_FLOAT, 0, vertices);
		glNormalPointer(GL_FLOAT, 0, normals);
		glColorPointer(3, GL_FLOAT, 0, colors);

		// Draw elements
		glDrawElements(GL_TRIANGLES, 3 * number_of_faces, GL_UNSIGNED_INT, m_indices);

		glEndList();

	}

}

void StaticMesh::load(string filename){

	PLYIO w;
	w.read(filename);

	size_t n_normals = 0;
	size_t n_colors = 0;

	unsigned int* tmp_ind;

	vertices  = w.getVertexArray(number_of_vertices);
	normals = w.getNormalArray(n_normals);
	colors = w.getColorArray(n_colors);
	m_indices = w.getIndexArray(number_of_faces);


	if(n_colors == 0)
	{
		colors = new float[number_of_vertices * 3];
		for(size_t i = 0; i < number_of_vertices; i++)
		{
			colors[i * 3] = 0.0;
			colors[i * 3 + 1] = 1.0;
			colors[i * 3 + 2] = 0.0;
		}

	}

	if(n_normals == 0)
	{
		interpolateNormals();
	}

	finalized = true;
	compileDisplayList();

}

void StaticMesh::interpolateNormals()
{
	// Be sure that vertex and indexbuffer exist
	assert(vertices);
	assert(m_indices);

	// Alloc new normal array
	normals = new float[3 * number_of_vertices];
	memset(normals, 0, 3 * number_of_vertices * sizeof(float));

	// Interpolate surface normals for each face
	// and interpolate sum up the normal coordinates
	// at each vertex position
	unsigned int a, b, c, buffer_pos;
	for(size_t i = 0; i < number_of_faces; i++)
	{
		buffer_pos = i * 3;

		// Interpolate a perpendicular vector to the
		// current triangle (p)
		//
		// CAUTION:
		// --------
		// buffer_pos is the face number
		// to get real position of the vertex in the buffer
		// we have to remember, that each vertex has three
		// coordinates!
		a = m_indices[buffer_pos]     * 3;
		b = m_indices[buffer_pos + 1] * 3;
		c = m_indices[buffer_pos + 2] * 3;

		Vertex v0(vertices[a], vertices[a + 1], vertices[a + 2]);
		Vertex v1(vertices[b], vertices[b + 1], vertices[b + 2]);
		Vertex v2(vertices[c], vertices[c + 1], vertices[c + 2]);

		Vertex d1 = v0 - v1;
		Vertex d2 = v2 - v1;

		Normal p(d1.cross(d2));

		// Sum up coordinate values in normal array
		normals[a    ] = p.x;
		normals[a + 1] = p.y;
		normals[a + 2] = p.z;

		normals[b    ] = p.x;
		normals[b + 1] = p.y;
		normals[b + 2] = p.z;

		normals[c    ] = p.x;
		normals[c + 1] = p.y;
		normals[c + 2] = p.z;

	}

	// Normalize
	for(size_t i = 0; i < number_of_vertices; i++)
	{
		Normal n(normals[i * 3], normals[i * 3 + 1], normals[i * 3 + 2]);
		normals[i * 3]     = n.x;
		normals[i * 3 + 1] = n.y;
		normals[i * 3 + 2] = n.z;
	}

}

void StaticMesh::save(string filename){

	if(finalized){

		cout << "Static mesh:: save" << endl;

		PLYIO ply_writer;

		// Create element descriptions
		PLYElement* vertex_element = new PLYElement("vertex", number_of_vertices);
		vertex_element->addProperty("x", "float");
		vertex_element->addProperty("y", "float");
		vertex_element->addProperty("z", "float");

		PLYElement* face_element = new PLYElement("face", number_of_faces);
		face_element->addProperty("vertex_indices", "uint", "uchar");


		// Add elements descriptions to header
		ply_writer.addElement(vertex_element);
		ply_writer.addElement(face_element);

		// Set data arrays
		ply_writer.setVertexArray(vertices, number_of_vertices);
		ply_writer.setIndexArray(m_indices, number_of_faces);

		// Save
		ply_writer.save(filename, true);

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
