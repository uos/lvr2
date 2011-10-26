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
 * BaseMesh.h
 *
 *  Created on: 03.02.2011
 *      Author: Thomas Wiemann
 */

#ifndef BASEMESH_H_
#define BASEMESH_H_

#include "io/MeshIO.hpp"

namespace lssr {

/**
 * @brief 	Abstract interface class for dynamic triangle meshes.
 * 			The surface reconstruction algorithm can handle all
 *			all data structures that allow sequential insertion
 *			all of indexed triangles.
 */
template<typename VertexT, typename NormalT>
class BaseMesh
{
public:

	BaseMesh();

	/**
	 * @brief 	This method should be called every time
	 * 			a new vertex is created.
	 *
	 * @param	v 		A supported vertex type. All used vertex types
	 * 					must support []-access.
	 */
	virtual void addVertex(VertexT v) = 0;

	/**
	 * @brief 	This method should be called every time
	 * 			a new vertex is created to ensure that vertex
	 * 			and normal buffer always have the same size
	 *
	 * @param	n 		A supported vertex type. All used vertex types
	 * 					must support []-access.
	 */
	virtual void addNormal(NormalT n) = 0;

	/**
	 * @brief 	Insert a new triangle into the mesh
	 *
	 * @param	a 		The first vertex of the triangle
	 * @param 	b		The second vertex of the triangle
	 * @param	c		The third vertex of the triangle
	 */
	virtual void addTriangle(uint a, uint b, uint c) = 0;

	/**
	 * @brief 	Finalizes a mesh, i.e. converts the template based buffers
	 * 			to OpenGL compatible buffers
	 */
	virtual void finalize() = 0;

	/**
	 * @brief Save the mesh to the given file
	 */
	virtual void save(string filename);

	/**
	 * @brief Save the mesh to the given Obj file
	 */
	virtual void saveObj(string filename);


	MeshIO* getMeshLoader();

protected:

	/// True if mesh is finalized
	bool			m_finalized;

	/// The mesh's vertex buffer
	float*			m_vertexBuffer;

	/// The vertex normals
	float*			m_normalBuffer;

	/// The vertex colors
	uchar*			m_colorBuffer;

	/// The texture coordinates
	float*			m_textureCoordBuffer;

	/// The texture indices
	uint*			m_textureIndexBuffer;

	/// The mesh's index buffer
	uint*			m_indexBuffer;

	/// The mesh's texture numbers
	uint*			m_textureBuffer;

	/// The number of vertices in the mesh
	uint			m_nVertices;

	/// The number of the mesh's texture numbers
	uint			m_nTextures;

	/// The number of face in the mesh
	uint 			m_nFaces;
};
}

#include "BaseMesh.tcc"

#endif /* BASEMESH_H_ */
