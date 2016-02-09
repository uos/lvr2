/* Co1pyright (C) 2011 Uni Osnabrück
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
 *  @author Sebastian Pütz (spuetz@uos.de)
 */

#ifndef FUSIONVERTEX_H_
#define FUSIONVERTEX_H_

#include <vector>
using namespace std;

#include "HalfEdgeVertex.hpp"

namespace lvr
{

/**
 * @brief	A vertex in a fusion mesh
 */
template<typename VertexT, typename NormalT>
class FusionVertex : public HalfEdgeVertex<VertexT, NormalT>
{
public:

	/**
	 * @brief	Default ctor. Position is (0, 0, 0), normal is undefined
	 */
		
	void init()
	{
		m_self_index = -1;
		is_valid = false;
	}
	
	/**
	 * @brief	Default ctor. Position is (0, 0, 0), normal is undefined
	 */
	FusionVertex() 
	: HalfEdgeVertex<VertexT, NormalT>::HalfEdgeVertex()
	{
		init();
	}

	/**
	 * @brief	Constructs a vertex at given position
	 */
	FusionVertex(VertexT vertex)
		: HalfEdgeVertex<VertexT, NormalT>(vertex)
	{
		init();
	}

	/**
	 * @brief	Copy Ctor.
	 */
	FusionVertex(const FusionVertex& o)
		: HalfEdgeVertex<VertexT, NormalT>::HalfEdgeVertex(o)
	{
		m_tree_dist = o.m_tree_dist;
		is_valid = o.is_valid;
		m_self_index = o.m_self_index;
	}

	/// The vertex index in the global mesh
	size_t 				m_self_index;
	
	/// The vertex distance to the tree representing the global buffer
	double 			m_tree_dist;
	
	/// Indicator if the vertex is valid or supposed to be deleted
	bool 				is_valid;
};

} // namespace lvr

#endif /* FUSIONVERTEX_H_ */
