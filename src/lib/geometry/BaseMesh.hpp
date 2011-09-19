/*
 * BaseMesh.h
 *
 *  Created on: 03.02.2011
 *      Author: Thomas Wiemann
 */

#ifndef BASEMESH_H_
#define BASEMESH_H_

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


protected:

	/// True if mesh is finalized
	bool			m_finalized;

	/// The mesh's vertex buffer
	float*			m_vertexBuffer;

	/// The vertex normals
	float*			m_normalBuffer;

	/// The vertex colors
	float*			m_colorBuffer;

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
