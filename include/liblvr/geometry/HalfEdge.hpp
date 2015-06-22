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
 * HalfEdge.hpp
 *
 * @author Thomas Wiemann (twiemann@uos.de)
 */

 #ifndef __HALF_EDGE_H__
#define __HALF_EDGE_H__

#include <boost/shared_ptr.hpp>

namespace lvr
{

/**
 * @brief An edge representation in a half edge mesh.
 */
template<typename HVertexT, typename FaceT>
struct HalfEdge{
public:

    /**
     * @brief   Ctor.
     */
	HalfEdge() : used(false), s(0), e(0), f(0), n(0), p(0) {};

	HalfEdge<HVertexT, FaceT>*  next();
	HalfEdge<HVertexT, FaceT>*  pair();
	HVertexT*                   start();
	HVertexT*                   end();
	FaceT*                      face();

	void setNext  (HalfEdge<HVertexT, FaceT>* next)    { n = next;};
	void setPair  (HalfEdge<HVertexT, FaceT>* pair)    { p = pair;};
	void setStart (HVertexT* start)   {s = start;};
	void setEnd   (HVertexT* end)     {e = end;};
	void setFace  (FaceT* face)       {f = face;};

	bool isBorderEdge();
	bool hasNeighborFace();
	bool hasFace();
	bool hasPair();

private:
	/// A pointer to the next edge in current contour
	HalfEdge<HVertexT, FaceT>* n;

	/// A pointer to the pair edge of this edge
	HalfEdge<HVertexT, FaceT>* p;

	/// A pointer to the start vertex of this edge
	HVertexT* s;

	/// A pointer to the end vertex of this edge
	HVertexT* e;

	/// A pointer to the surrounded face
	FaceT* f;

public:
	/// Used for clustering (may be removed soon)
	bool used;
};


} // namespace lvr

#include "HalfEdge.tcc"

#endif
