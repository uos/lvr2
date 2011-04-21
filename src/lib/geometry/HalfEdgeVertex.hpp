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
	vector<HalfEdge< VertexT, HalfEdgeFace<VertexT, NormalT> > *> in;

	/// The list of outgoing edges
	vector<HalfEdge< VertexT, HalfEdgeFace<VertexT, NormalT> > *> out;
};

} // namespace lssr

#endif /* HALFEDGEVERTEX_H_ */
