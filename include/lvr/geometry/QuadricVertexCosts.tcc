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
 * QuadricVertexCosts.cpp
 *
 *  Created on: 22.04.2012
 *      Author: Thomas Wiemann
 */


namespace lvr
{

template<typename VertexT, typename NormalT>
float QuadricVertexCosts<VertexT, NormalT>::operator()(HalfEdgeVertex<VertexT, NormalT> &v)
{
	float mincost = std::numeric_limits<float>::max();
	bool hasNeighbors = false;

	Matrix4<float> q1;
	v.calcQuadric(q1, false);

	// Iterator over all neighbour vertices
	typename vector<HEdge* >::iterator it;
	for (it = v.out.begin(); it != v.out.end(); it++)
	{

		HVertex* n = (*it)->end();

		Matrix4<float> q2;

		// Add two 4x4 Q matrices
		n->calcQuadric(q2, false);

		Matrix4<float> qsum = q1 + q2;

		/*double triArea = 0;
			if (QUADRICTRI == _cost)
			{
				triArea = v.getQuadricSummedTriArea() + n.getQuadricSummedTriArea();
			}*/

		float triArea = 0;
		// calc cost
		float cost = calcQuadricError(qsum, n, triArea);

		if (cost < mincost)
		{
			hasNeighbors = true;
			mincost = cost;
		}
	}

	if (hasNeighbors)
	{
		return mincost;
	}
	else
	{
		return FLT_MAX; // vertex not connected to an edge
	}
}

template<typename VertexT, typename NormalT>
float QuadricVertexCosts<VertexT, NormalT>::calcQuadricError(Matrix4<float> &quadric, HVertex* v, float area)
{
	float cost;

	// 1st, consider vertex v a 1x4 matrix: [v.x v.y v.z 1]
	// Multiply it by the Qsum 4x4 matrix, resulting in a 1x4 matrix

	float result[4];

	VertexT v3 = v->m_position;

	result[0] = v3.x * quadric[0 ] + v3.y * quadric[1 ] + v3.z * quadric[2 ] + 1 * quadric[3 ];
	result[1] = v3.x * quadric[4 ] + v3.y * quadric[5 ] + v3.z * quadric[6 ] + 1 * quadric[7 ];
	result[2] = v3.x * quadric[8 ] + v3.y * quadric[9 ] + v3.z * quadric[10] + 1 * quadric[11];
	result[3] = v3.x * quadric[12] + v3.y * quadric[13] + v3.z * quadric[14] + 1 * quadric[15];

	// Multiply this 1 x 4 matrix by the vertex v transpose (a 4 x 1 matrix).
	// This is just the dot product.

	cost =	result[0] * v3.x + result[1] * v3.y + result[2] * v3.z + result[3] * 1;

	if (m_useTri && area != 0)
	{
		cost /= area;
	}

	return cost;
}


} /* namespace lvr */
