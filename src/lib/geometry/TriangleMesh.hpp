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
 * TriangleMesh.hpp
 *
 *  @date 17.10.2008
 *  @author Thomas Wiemann (twiemann@uos.de)
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
