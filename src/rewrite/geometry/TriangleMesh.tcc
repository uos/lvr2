/*
 * TriangleMesh.cpp
 *
 *  Created on: 17.10.2008
 *      Author: Thomas Wiemann
 */

#include "../io/PLYIO.hpp"

namespace lssr
{

template<typename VertexType, typename IndexType>
TriangleMesh<VertexType, IndexType>::TriangleMesh()
{

}

template<typename VertexType, typename IndexType>
TriangleMesh<VertexType, IndexType>::~TriangleMesh<VertexType, IndexType>()
{
    m_normals.clear();
    m_vertices.clear();
    m_triangles.clear();
}

template<typename VertexType, typename IndexType>
void TriangleMesh<VertexType, IndexType>::addTriangle(IndexType v0, IndexType v1, IndexType v2)
{
	// Check indices
	assert(v0 < m_vertices.size());

	// Insert the vertex indices into the index buffer
	m_triangles.push_back(v0);
	m_triangles.push_back(v1);
	m_triangles.push_back(v2);

	// Check if normals were created together with vertices.
	// If this was done consistently, normal and vertex
	// buffer should have the same size.
	if(m_normals.size() == m_vertices.size())
	{

		// If we store normals, interpolate the current face
		// normal with the appropriate vertex normals
		VertexType diff1 = m_vertices[v0] - m_vertices[v1];
		VertexType diff2 = m_vertices[v1] - m_vertices[v2];
		VertexType normal = diff1.cross(diff2);

		// Interpolate the vertex normals (i.e. sum them up).
		// Normalization is done after the complete mesh
		// was has been created
		m_normals[v0] += normal;
		m_normals[v1] += normal;
		m_normals[v2] += normal;
	}
}

template<typename VertexType, typename IndexType>
VertexType TriangleMesh<VertexType, IndexType>::getVertex(IndexType index)
{
	assert(index < (int) m_vertices.size());
	return m_vertices[index];
}

template<typename VertexType, typename IndexType>
VertexType TriangleMesh<VertexType, IndexType>::getNormal(IndexType index)
{
	assert(index < (int)m_normals.size());
	return m_normals[index];
}

template<typename VertexType, typename IndexType>
void TriangleMesh<VertexType, IndexType>::finalize()
{
	// Alloc buffers
	this->m_vertexBuffer = new float[3 * m_vertices.size()];
	this->m_normalBuffer = new float[3 * m_vertices.size()];
	this->m_indexBuffer = new uint[m_triangles.size()];

	// Fill buffers
	int index = 0;
	for(size_t i = 0; i < m_vertices.size(); i++)
	{
		index = 3 * i;
		this->m_vertexBuffer[index    ] = (float)m_vertices[i][0];
		this->m_vertexBuffer[index + 1] = (float)m_vertices[i][1];
		this->m_vertexBuffer[index + 2] = (float)m_vertices[i][2];

		this->m_normalBuffer[index    ] = (float)m_normals[i][0];
		this->m_normalBuffer[index + 1] = (float)m_normals[i][1];
		this->m_normalBuffer[index + 2] = (float)m_normals[i][2];
	}

	int c = 0;
	for(typename list<IndexType>::iterator it = m_triangles.begin();
		it != m_triangles.end(); it++)
	{
		this->m_indexBuffer[c] = (unsigned int)*it;
		c++;
	}

	this->m_finalized = true;
}

template<typename VertexType, typename IndexType>
void TriangleMesh<VertexType, IndexType>::save(string filename)
{
	PLYIO ply_writer;

	// Create element descriptions
	PLYElement* vertex_element = new PLYElement("vertex", m_vertices.size());
	vertex_element->addProperty("x", "float");
	vertex_element->addProperty("y", "float");
	vertex_element->addProperty("z", "float");

	PLYElement* face_element = new PLYElement("face", m_triangles.size());
	face_element->addProperty("vertex_indices", "uint", "uchar");


	// Add elements descriptions to header
	ply_writer.addElement(vertex_element);
	ply_writer.addElement(face_element);

	// Set data arrays
	ply_writer.setVertexArray(this->m_vertexBuffer, this->m_vertices.size());
	ply_writer.setIndexArray(this->m_indexBuffer, m_triangles.size() / 3);

	// Save
	ply_writer.save(filename, true);
}

/// TODO: Re-Integrate normal interpolation
//template<typename VertexType, typename IndexType>
//void TriangleMesh<VertexType, IndexType>::interpolateNormal(Normal n, size_t index){
//
//	if(index < normal_buffer.size()){
//
//		Normal normal = normal_buffer[index];
//		normal += n;
//		normal.normalize();
//
//		normal_buffer[index] = normal;
//
//	} else {
//		cout << "WARNING: TriangleMesh<VertexType, IndexType>: Normal index out of range: " << index << endl;
//	}
//
//}


}
