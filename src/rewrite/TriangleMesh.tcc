/*
 * TriangleMesh.cpp
 *
 *  Created on: 17.10.2008
 *      Author: Thomas Wiemann
 */

#include "PLYIO.hpp"

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
		// Normalization is don after the complete mesh
		// was has been created
		m_vertices[v0] += normal;
		m_vertices[v1] += normal;
		m_vertices[v2] += normal;
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
void TriangleMesh<VertexType, IndexType>::save(string filename)
{
//	PLYIO ply_writer;
//
//	// Create element descriptions
//	PLYElement* vertex_element = new PLYElement("vertex", m_vertices.size());
//	vertex_element->addProperty("x", "float");
//	vertex_element->addProperty("y", "float");
//	vertex_element->addProperty("z", "float");
//
//	PLYElement* face_element = new PLYElement("face", m_triangles.size());
//	face_element->addProperty("vertex_indices", "uint", "uchar");
//
//
//	// Add elements descriptions to header
//	ply_writer.addElement(vertex_element);
//	ply_writer.addElement(face_element);
//
//	// Set data arrays
//	ply_writer.setVertexArray(m_vertices, m_vertices.size());
//	ply_writer.setIndexArray(m_triangles, m_triangles.size());
//
//	// Save
//	ply_writer.save(filename, true);
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
