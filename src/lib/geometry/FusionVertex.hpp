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
 * FusionVertex.hpp
 *
 *  @date 14.07.2013
 *  @author Henning Deeken (hdeeken@uos.de)
 */

#ifndef FUSIONVERTEX_H_
#define FUSIONVERTEX_H_

#include <vector>
using namespace std;

#include "Vertex.hpp"
#include "Normal.hpp"
//#include "FusionEdge.hpp"
//#include "FusionFace.hpp"

namespace lssr
{

/**
 * @brief	A vertex in a fusion mesh
 */
template<typename VertexT, typename NormalT> class FusionVertex
{
	
public:

	//typedef FusionEdge<FusionVertex<VertexT, NormalT>, FusionFace<VertexT, NormalT> > FEdge;
	//typedef FusionFaceFace<VertexT, NormalT> FFace;

	/**
	 * @brief	Default ctor. Position is (0, 0, 0), normal is undefined
	 */
	FusionVertex() 
	{
		m_self_index = 1337;
	}

	/**
	 * @brief	Constructs a vertex at given position with provided normal.
	 */
	FusionVertex(VertexT v, NormalT n) : m_position(v), m_normal(n) {}

	/**
	 * @brief	Constructs a vertex at given position
	 */
	FusionVertex(VertexT v) : m_position(v) {}

	/**
	 * @brief	Copy Ctor.
	 */
	 /*
	FusionVertex(const FusionVertex& o)
	{
		m_position = o.m_position;
		m_normal = o.m_normal;
		m_self_index = o.m_self_index;
		is_border_vertex = o.is_border_vertex;
	}
	*/

	/// The vertex's position
	VertexT 			m_position;

	/// The vertex's normal
	NormalT 			m_normal;

	/// The vertex index in the mesh
	size_t 				m_self_index;
	
	/// The vertex index in the mesh
	//vector<int> 		m_face_indices;

	/// Indicator if vertex is part of the mesh boundary
	bool is_border_vertex;

};

} // namespace lssr

#endif /* FUSIONVERTEX_H_ */
