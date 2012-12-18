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
 * VertexCosts.h
 *
 *  Created on: 22.04.2012
 *      Author: Thomas Wiemann
 */

#ifndef VERTEXCOSTS_H_
#define VERTEXCOSTS_H_

#include <limits>

#include "HalfEdgeVertex.hpp"

namespace lssr
{

/**
 * @brief Base class cost determination of vertex removals. Re-Implement the
 *        operator()-method to define a cost function
 */
template<typename VertexT, typename NormalT>
class VertexCosts
{
public:
	VertexCosts() {};

	/**
	 * @brief	Implementation of the vertex cost function. The default implementation
	 * 			returns the maximum float number. Using this method, no vertices
	 * 			should be removed from the mesh.
	 */
	virtual float operator()(HalfEdgeVertex<VertexT, NormalT> &v) { return std::numeric_limits<float>::max(); }
};



typedef std::pair<HalfEdgeVertex<ColorVertex<float, uchar>, Normal<float> >*, float> vertexCost_p;

struct cVertexCost
{
	bool operator()(vertexCost_p &p1, vertexCost_p &p2)
	{
		return p1.second < p2.second;
	}
};


} /* namespace lssr */
#endif /* VERTEXCOSTS_H_ */
