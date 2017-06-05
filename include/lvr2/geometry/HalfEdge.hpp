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
 *  @date 02.06.2017
 *  @author Lukas Kalbertodt <lukas.kalbertodt@gmail.com>
 */

#ifndef LVR2_GEOMETRY_HALFEDGE_H_
#define LVR2_GEOMETRY_HALFEDGE_H_


namespace lvr2
{

// Forward definitions
template <typename VectorT> struct HalfEdgeFace;
template <typename VectorT> struct HalfEdgeVertex;

template <typename VectorT>
struct HalfEdge
{
    using Edge = HalfEdge<VectorT>;
    using Face = HalfEdgeFace<VectorT>;
    using Vertex = HalfEdgeVertex<VectorT>;


    /// The face this edge belongs to.
    Face* face;

    /// The vertex this edge points to.
    Vertex* target;

    /// The next edge of the face, ordered counter-clockwise. Is equal to `this->target->outgoing`.
    Edge* next;

    /// The pair edge.
    Edge* pair;
};

} // namespace lvr

#endif /* LVR2_GEOMETRY_HALFEDGE_H_ */
