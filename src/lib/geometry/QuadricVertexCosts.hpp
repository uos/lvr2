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
 * QuadricVertexCosts.h
 *
 *  Created on: 22.04.2012
 *      Author: Thomas Wiemann
 */

#ifndef QUADRICVERTEXCOSTS_H_
#define QUADRICVERTEXCOSTS_H_

#include "VertexCosts.hpp"

namespace lssr
{

/**
 * @brief	Implements Garland & Heckbert's quadric based vertex
 * 			removal cost function. Implementation is based on Jeff
 * 			Somers source code
 */
template<typename VertexT, typename NormalT>
class QuadricVertexCosts : public VertexCosts<VertexT, NormalT>
{
	typedef HalfEdgeVertex<VertexT, NormalT> HVertex;
	typedef HalfEdge<HVertex, HalfEdgeFace<VertexT, NormalT> > HEdge;

public:

	/**
	 * Ctor.
	 *
	 * @param useTriangleArea 	If true, the areas of the surrounding triangles
	 *    						will be taken into account for error calculation
	 */
	QuadricVertexCosts(bool useTriangleArea) : m_useTri(useTriangleArea) {};

	/**
	 * @brief 	Implementation of Garland and Heckberts cost function. If the object
	 * 			was created with the useTriangleArea flag, the weighted costs function
	 * 			will be used.
	 */
	virtual float operator()(HalfEdgeVertex<VertexT, NormalT> &v);

private:

	float calcQuadricError(Matrix4<float> &quadric, HVertex* v, float area);

	bool m_useTri;

};

} /* namespace lssr */

#include "QuadricVertexCosts.tcc"

#endif /* QUADRICVERTEXCOSTS_H_ */
