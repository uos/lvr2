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
 * HalfEdgeVertex.tcc
 *
 *  Created on: 22.04.2012
 *      Author: Thomas Wiemann
 */

namespace lssr
{

template<typename VertexT, typename NormalT>
void HalfEdgeVertex<VertexT, NormalT>::calcQuadric(Matrix4<float> &q, bool use_tri)
{
	// Get adjacent faces
	list<HFace*> adj_faces;
	getAdjacentFaces(adj_faces);

	// Init quadric data
	float* data = q.getData();
	for(int i = 0; i < 16; i++) data[i] = 0;

	// Calculate quadric entries
	typename list<HFace*>::iterator it;
	for(it = adj_faces.begin(); it != adj_faces.end(); it++)
	{
		HFace* f = *it;

		float triangle_area = 1;
		if(use_tri)
		{
			triangle_area = f->getArea();
		}

		NormalT n = f->getFaceNormal();

		float a = n[0];
		float b = n[1];
		float c = n[2];
		float d = f->getD();

		data[0] += triangle_area * a * a;
		data[1] += triangle_area * a * b;
		data[2] += triangle_area * a * c;
		data[3] += triangle_area * a * d;

		data[4] += triangle_area * b * a;
		data[5] += triangle_area * b * b;
		data[6] += triangle_area * b * c;
		data[7] += triangle_area * b * d;

		data[8] += triangle_area * c * a;
		data[9] += triangle_area * c * b;
		data[10] += triangle_area * c * c;
		data[11] += triangle_area * c * d;

		data[12] += triangle_area * d * a;
		data[13] += triangle_area * d * b;
		data[14] += triangle_area * d * c;
		data[15] += triangle_area * d * d;

	}
}


template<typename VertexT, typename NormalT>
void HalfEdgeVertex<VertexT, NormalT>::getAdjacentFaces(list<HalfEdgeFace<VertexT, NormalT>* > &adj )
{
	set<HFace*> adj_faces;
	typename vector<HEdge*>::iterator it;

	// Iterate over incoming edges and get all surrounding faces.
	// In a correctly linked  mesh it shouldn't be necessary to
	// iterate over the outgoint edges as well.

	for(it = out.begin(); it != out.end(); it++)
	{
		HEdge* e = *it;
		if(e->face)
		{
			adj_faces.insert(e->face);
		}

		if(e->pair)
		{
			if(e->pair->face)
			{
				adj_faces.insert(e->pair->face);
			}
		}
	}

	// Copy pointers to out list
	typename set<HFace*>::iterator sit;
	for(sit = adj_faces.begin(); sit != adj_faces.end(); sit++)
	{
		adj.push_back(*sit);
	}
}

template<typename VertexT, typename NormalT>
bool HalfEdgeVertex<VertexT, NormalT>::isBorderVertex()
{
	list<HalfEdgeFace<VertexT, NormalT>* > adj;
	typename list<HalfEdgeFace<VertexT, NormalT>* >::iterator it;
	getAdjacentFaces(adj);

	for(it = adj.begin(); it != adj.end(); it++)
	{
		HalfEdgeFace<VertexT, NormalT>* f = *it;
		if(f->isBorderFace())
		{
			return false;
		}
	}
	return true;
}

template<typename VertexT, typename NormalT>
HalfEdge< HalfEdgeVertex<VertexT, NormalT>, HalfEdgeFace<VertexT, NormalT> >*  HalfEdgeVertex<VertexT, NormalT>::getShortestEdge()
{
	HEdge* shortest = 0;
	float s_length = numeric_limits<float>::max();

	typename vector<HEdge*>::iterator it;
	for(it = in.begin(); it != in.end(); it++)
	{
		HEdge* e = *it;
		VertexT v1 = e->start->m_position;
		VertexT v2 = e->end->m_position;
		float length = (v1 - v2).length();

		if(shortest == 0 || length < s_length )
		{
			s_length = length;
			shortest = e;
		}
	}
	return shortest;
}

} // namespace lssr
