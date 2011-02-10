/*
 * TriangleMesh.h
 *
 *  Created on: 03.02.2011
 *      Author: Thomas Wiemann
 */

#ifndef INDEXED_TRIANGLEMESH_H_
#define INDEXED_TRIANGLEMESH_H_

#include "BaseMesh.hpp"
#include "Vertex.hpp"

#include <list>

using std::list;

namespace lssr
{

/**
 * @brief 	An implementation of an indexed triangle mesh.
 */
template <typename VertexType, typename IndexType>
class IndexedTriangleMesh : public BaseMesh<VertexType, IndexType>
{
public:

	/**
	 * @brief	Default constructor. Initializes an empty mesh.
	 */
	IndexedTriangleMesh() {};

	/**
	 * @brief	Inserts a new vertex into the mesh
	 */
	virtual void addVertex(VertexType v)
	{
		m_vertexList.push_back(v);
	}

	/**
	 * @brief	Inserts a new triangle consisting of a, b, c
	 * 			into the mesh
	 */
	virtual void addTriangle(IndexType a, IndexType b, IndexType c)
	{
		m_indexList.push_back(a);
		m_indexList.push_back(b);
		m_indexList.push_back(c);
	}

	/**
	 * @brief	Destructor.
	 */
	virtual ~IndexedTriangleMesh() {};

protected:

	/**
	 * @brief	The vertex list of the mesh
	 */
	list<VertexType>		m_vertexList;

	/**
	 * @brief 	The vertex index list for the triangles in the mesh.
	 */
	list<IndexType>			m_indexList;
};

typedef IndexedTriangleMesh<Vertexf, size_t> TriangleMesh;

}

#endif
