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
 * FusionFace.cpp
 *
 *  @date 03.12.2008
 *  @author Henning Deeken (hdeeken@uos.de)
 *  @author Ann-Katrin Häuser (ahaeuser@uos.de)
 *  @author Sebastian Pütz (spuetz@uos.de)
 */

#include "FusionFace.hpp"
#include "FusionVertex.hpp"

// Forward declaration
template<typename A, typename B> class FusionVertex;

namespace lvr
{

	template<typename VertexT, typename NormalT> FusionFace<VertexT, NormalT>::FusionFace() 
		: HalfEdgeFace<VertexT, NormalT>()
	{
		// color
		r = 0;
		g = 255;
		b = 0;

		m_self_index = -1;
		is_valid = false;
	}

	template<typename VertexT, typename NormalT> FusionFace<VertexT, NormalT>::~FusionFace() {}


} // namespace lvr

