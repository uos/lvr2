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
 * HalfEdgeVertex.h
 *
 *  Created on: 03.12.2008
 *      Author: twiemann
 */

#ifndef HALFEDGEVERTEX_H_
#define HALFEDGEVERTEX_H_

#include <vector>
using namespace std;

#include "Vertex.hpp"
#include "Normal.hpp"
#include "HalfEdge.hpp"
#include "HalfEdgeFace.hpp"

namespace lssr
{

/**
 * @brief	A vertex in a half edge mesh
 */
template<typename VertexT, typename NormalT>
class HalfEdgeVertex
{
public:

	/**
	 * @brief	Default ctor. Position is (0, 0, 0), normal is undefined
	 */
	HalfEdgeVertex() {}

	/**
	 * @brief	Constructs a vertex at given position with provided normal.
	 */
	HalfEdgeVertex(VertexT v, NormalT n) : m_position(v), m_normal(n) {}

	/**
	 * @brief	Constructs a vertex at given position
	 */
	HalfEdgeVertex(VertexT v) : m_position(v) {}

	/**
	 * @brief	Copy Ctor.
	 */
	HalfEdgeVertex(const HalfEdgeVertex& o)
	{
		m_position = o.m_position;
		m_normal = o.m_normal;
		m_index = o.m_index;
	}

	/// The vertex's position
	VertexT 			m_position;

	/// The vertex's normal
	NormalT 			m_normal;

	/// The vertex index in the mesh
	size_t 				m_index;

	/// The list incoming edges
	vector<HalfEdge< HalfEdgeVertex<VertexT, NormalT>, HalfEdgeFace<VertexT, NormalT> > *> in;

	/// The list of outgoing edges
	vector<HalfEdge< HalfEdgeVertex<VertexT, NormalT>, HalfEdgeFace<VertexT, NormalT> > *> out;
};

} // namespace lssr

#endif /* HALFEDGEVERTEX_H_ */
