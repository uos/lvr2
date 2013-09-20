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
	HalfEdge() : next(0), pair(0), start(0), end(0), face(0), used(false) {};

	~HalfEdge(){
	
	}

	/// A pointer to the next edge in current contour
	HalfEdge<HVertexT, FaceT>* next;

	/// A pointer to the pair edge of this edge
	HalfEdge<HVertexT, FaceT>* pair;

	/// A pointer to the start vertex of this edge
	HVertexT* start;

	/// A pointer to the end vertex of this edge
	HVertexT* end;

	/// A pointer to the surrounded face
	FaceT* face;

	/// Used for clustering (may be removed soon)
	bool used;
};

} // namespace lvr

#endif
