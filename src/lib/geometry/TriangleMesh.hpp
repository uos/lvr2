/*
 * TriangleMesh.h
 *
 *  Created on: 17.10.2008
 *      Author: Thomas Wiemann
 */

#ifndef TRIANGLEMESH_H_
#define TRIANGLEMESH_H_

#include "BaseMesh.hpp"

#include <vector>
#include <list>
#include <cassert>
using namespace std;

namespace lssr
{

/**
 * @brief An implementation of an indexed triangle mesh
 */
template<typename VertexT, typename NormalT>
class TriangleMesh : public BaseMesh<VertexT, NormalT>{
public:

	/**
	 * @brief Constructor.
	 */
	TriangleMesh();

	/**
	 * @brief Copy ctor.
	 */
	TriangleMesh(const TriangleMesh &other);

	/**
	 * @brief Adds a triangle consisting of the given vertices (by
	 * 		  by their index) into the mesh
	 *
	 * @param v0		The first vertex index
	 * @param v1		The second vertex index
	 * @param v2		The third vertex index
	 */
	virtual void addTriangle(uint v0, uint v1, uint v2);

	/**
	 * @brief Adds a new vertex into the mesh
	 *
	 * @param v			The new vertex
	 */
	virtual void addVertex(VertexT v){ m_vertices.push_back(v);};

	/**
	 * @brief Adds a new normal into the mesh. Vertices and normals
	 * 		  should always by corresponding wrt. their indices.
	 *
	 * @param n 		The new normal
	 */
	virtual void addNormal(NormalT n) {m_normals.push_back(n);};


	/**
	 * @brief Dtor.
	 */
	virtual ~TriangleMesh();

	/**
	 * @brief Returns the vertex at the given index
	 */
	VertexT   getVertex(uint index);

	/**
	 * @brief Returns the normal at the given index
	 */
	VertexT   getNormal(uint index);

	/**
	 * @brief Finalizes the mesh.
	 */
	virtual void finalize();

protected:

	/// An interlaced buffer for vertex normals
	float*					m_normalBuffer;

	/// The vertex normals
	vector<VertexT>         m_normals;

	/// The vertices
	vector<VertexT>         m_vertices;

	/// The index list
	list<uint>              m_triangles;

};

} // namepsace lssr

#include "TriangleMesh.tcc"

#endif /* TRIANGLEMESH_H_ */
