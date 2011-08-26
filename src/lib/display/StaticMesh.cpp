/*
 * StaticMesh.cpp
 *
 *  Created on: 12.11.2008
 *      Author: Thomas Wiemann
 */

#include "StaticMesh.hpp"

#include <cassert>

namespace lssr
{

StaticMesh::StaticMesh(){

	m_normals = 0;
	m_vertices = 0;
	m_colors = 0;
	m_indices = 0;

	m_numFaces = 0;
	m_numVertices = 0;

	m_finalized = false;

}

StaticMesh::StaticMesh(string name) : Renderable(name){

	m_normals = 0;
	m_vertices = 0;
	m_colors = 0;
	m_indices = 0;

	m_numFaces = 0;
	m_numVertices = 0;

	m_finalized = false;

	load(name);

}

StaticMesh::StaticMesh(const StaticMesh &o)
{

	if(m_normals != 0) delete[] m_normals;
	if(m_vertices != 0) delete[] m_vertices;
	if(m_colors != 0) delete[] m_colors;
	if(m_indices != 0) delete[] m_indices;

	m_normals = new float[3 * o.m_numVertices];
	m_vertices = new float[3 * o.m_numVertices];
	m_colors = new float[3 * o.m_numVertices];

	m_indices = new unsigned int[3 * o.m_numFaces];

	for(size_t i = 0; i < 3 * o.m_numVertices; i++)
	{
		m_normals[i] = o.m_normals[i];
		m_vertices[i] = o.m_vertices[i];
		m_colors[i] = o.m_colors[i];
	}

	for(size_t i = 0; i < 3 * o.m_numFaces; i++)
	{
		m_indices[i] = o.m_indices[i];
	}

	m_boundingBox = o.m_boundingBox;

}

StaticMesh::~StaticMesh(){

}

void StaticMesh::finalize(){

}

void StaticMesh::compileDisplayList(){

	if(m_finalized){

		m_listIndex = glGenLists(1);

		// Enable vertex / normal / color arrays
		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_NORMAL_ARRAY);
		glEnableClientState(GL_COLOR_ARRAY);

		// Start new display list
		glNewList(m_listIndex, GL_COMPILE);

		glEnable(GL_LIGHTING);

		// Assign element pointers
		glVertexPointer(3, GL_FLOAT, 0, m_vertices);
		glNormalPointer(GL_FLOAT, 0, m_normals);
		glColorPointer(3, GL_FLOAT, 0, m_colors);

		// Draw elements
		glDrawElements(GL_TRIANGLES, 3 * m_numFaces, GL_UNSIGNED_INT, m_indices);

		glEndList();

	}

}

void StaticMesh::load(string filename){

//	PLYIO w;
//	w.read(filename);
//
//	size_t n_m_normals = 0;
//	size_t n_colors = 0;
//
//	m_vertices  = w.getVertexArray(number_of_m_vertices);
//	m_normals = w.getNormalArray(n_m_normals);
//	colors = w.getColorArray(n_colors);
//	m_indices = w.getIndexArray(number_of_faces);
//
//	cout << m_normals << endl;
//
//	// Calculate bounding box
//	if(m_boundingBox) delete m_boundingBox;
//	m_boundingBox = new BoundingBox;
//
//	for(size_t i = 0; i < number_of_m_vertices; i++)
//	{
//		m_boundingBox->expand(m_vertices[3 * i], m_vertices[3 * i + 1], m_vertices[3 * i + 2]);
//	}
//
//	if(n_colors == 0)
//	{
//		colors = new float[number_of_m_vertices * 3];
//		for(size_t i = 0; i < number_of_m_vertices; i++)
//		{
//			colors[i * 3] = 0.0;
//			colors[i * 3 + 1] = 1.0;
//			colors[i * 3 + 2] = 0.0;
//		}
//	}
//
//	if(n_m_normals == 0)
//	{
//		interpolatem_normals();
//	}
//
////	for(int i = 0; i < number_of_m_vertices; i++)
////	{
////		m_vertices[3 * i] = m_vertices[3 * i] * 100;
////		m_vertices[3 * i + 1] = m_vertices[3 * i + 1] * 100;
////		m_vertices[3 * i + 2] = m_vertices[3 * i + 2] * 100;
////	}
//
//	finalized = true;
//	compileDisplayList();

}

void StaticMesh::interpolateNormals()
{
	// Be sure that vertex and indexbuffer exist
	assert(m_vertices);
	assert(m_indices);

	// Alloc new normal array
	m_normals = new float[3 * m_numVertices];
	memset(m_normals, 0, 3 * m_numVertices * sizeof(float));

	// Interpolate surface m_normals for each face
	// and interpolate sum up the normal coordinates
	// at each vertex position
	unsigned int a, b, c, buffer_pos;
	for(size_t i = 0; i < m_numFaces; i++)
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

		Vertex<float> v0(m_vertices[a], m_vertices[a + 1], m_vertices[a + 2]);
		Vertex<float> v1(m_vertices[b], m_vertices[b + 1], m_vertices[b + 2]);
		Vertex<float> v2(m_vertices[c], m_vertices[c + 1], m_vertices[c + 2]);

		Vertex<float> d1 = v0 - v1;
		Vertex<float> d2 = v2 - v1;

		Normal<float> p(d1.cross(d2));

		// Sum up coordinate values in normal array
		m_normals[a    ] = p.x;
		m_normals[a + 1] = p.y;
		m_normals[a + 2] = p.z;

		m_normals[b    ] = p.x;
		m_normals[b + 1] = p.y;
		m_normals[b + 2] = p.z;

		m_normals[c    ] = p.x;
		m_normals[c + 1] = p.y;
		m_normals[c + 2] = p.z;

	}

	// Normalize
	for(size_t i = 0; i < m_numVertices; i++)
	{
		Normal<float> n(m_normals[i * 3], m_normals[i * 3 + 1], m_normals[i * 3 + 2]);
		m_normals[i * 3]     = n.x;
		m_normals[i * 3 + 1] = n.y;
		m_normals[i * 3 + 2] = n.z;
	}

}

unsigned int* StaticMesh::getIndices()
{
	if(m_finalized)
	{
		return m_indices;
	}
	else
	{
		return 0;
	}
}

float* StaticMesh::getVertices()
{
	if(m_finalized)
	{
		return m_vertices;
	}
	else
	{
		return 0;
	}
}

float* StaticMesh::getNormals()
{
    if(m_finalized)
    {
        return m_normals;
    }
    else
    {
        return 0;
    }
}

size_t StaticMesh::getNumberOfVertices()
{
	return m_numVertices;
}
size_t StaticMesh::getNumberOfFaces()
{
	return m_numFaces;
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
//		if(m_vertices != 0)
//		{
//
//		}
//
//		if(m_normals != 0)
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

} // namespace lssr
