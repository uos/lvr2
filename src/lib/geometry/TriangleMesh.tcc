/* Copyright (C) 2011 Uni Osnabrück
 * This file is part of the LAS VEGAS Reconstruction Toolkit,
 *
 * LAS VEGAS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * LAS VEGAS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA
 */


/*
 * TriangleMesh.cpp
 *
 *  @date 17.10.2008
 *  @author Thomas Wiemann (twiemann@uos.de)
 */

#include "../io/PLYIO.hpp"

namespace lssr
{

template<typename VertexT, typename NormalT>
TriangleMesh<VertexT, NormalT>::TriangleMesh()
{

}

template<typename VertexT, typename NormalT>
TriangleMesh<VertexT, NormalT>::~TriangleMesh<VertexT, NormalT>()
{
    m_normals.clear();
    m_vertices.clear();
    m_triangles.clear();
}

template<typename VertexT, typename NormalT>
void TriangleMesh<VertexT, NormalT>::addTriangle(uint v0, uint v1, uint v2)
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
		VertexT diff1 = m_vertices[v0] - m_vertices[v1];
		VertexT diff2 = m_vertices[v1] - m_vertices[v2];
		VertexT normal = diff1.cross(diff2);

		// Interpolate the vertex normals (i.e. sum them up).
		// Normalization is done after the complete mesh
		// was has been created
		m_normals[v0] += normal;
		m_normals[v1] += normal;
		m_normals[v2] += normal;
	}
}

template<typename VertexT, typename NormalT>
VertexT TriangleMesh<VertexT, NormalT>::getVertex(uint index)
{
	assert(index < (int) m_vertices.size());
	return m_vertices[index];
}

template<typename VertexT, typename NormalT>
VertexT TriangleMesh<VertexT, NormalT>::getNormal(uint index)
{
	assert(index < (int)m_normals.size());
	return m_normals[index];
}

template<typename VertexT, typename NormalT>
void TriangleMesh<VertexT, NormalT>::finalize()
{
	// Alloc buffers
	this->m_vertexBuffer = new float[3 * m_vertices.size()];
	this->m_normalBuffer = new float[3 * m_vertices.size()];
	this->m_indexBuffer = new uint[m_triangles.size()];

	// Save primitive counts
	this->m_nFaces = m_triangles.size() / 3;
	this->m_nVertices = m_vertices.size();

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
	for(typename list<uint>::iterator it = m_triangles.begin();
		it != m_triangles.end(); it++)
	{
		this->m_indexBuffer[c] = (unsigned int)*it;
		c++;
	}

	this->m_finalized = true;
}

/// TODO: Re-Integrate normal interpolation
//template<typename VertexT, typename NormalT>
//void TriangleMesh<VertexT, uint>::interpolateNormal(Normal n, size_t index){
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
//		cout << "WARNING: TriangleMesh<VertexT, uint>: Normal index out of range: " << index << endl;
//	}
//
//}


}
