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
template <typename BaseVecT> struct HalfEdgeFace;
template <typename BaseVecT> struct HalfEdgeVertex;
template <typename BaseVecT> struct HalfEdgeMesh;

template <typename BaseVecT>
struct HalfEdge
{
    using Edge = HalfEdge<BaseVecT>;
    using Face = HalfEdgeFace<BaseVecT>;
    using Vertex = HalfEdgeVertex<BaseVecT>;

    using EdgeHandle = typename HalfEdgeMesh<BaseVecT>::EdgeHandle;
    using FaceHandle = typename HalfEdgeMesh<BaseVecT>::FaceHandle;
    using VertexHandle = typename HalfEdgeMesh<BaseVecT>::VertexHandle;

    /// The face this edge belongs to.
    FaceHandle face;

    /// The vertex this edge points to.
    VertexHandle target;

    /// The next edge of the face, ordered counter-clockwise. Is equal to `this->target->outgoing`.
    EdgeHandle next;

    /// The pair edge.
    EdgeHandle pair;
};

} // namespace lvr

#endif /* LVR2_GEOMETRY_HALFEDGE_H_ */
