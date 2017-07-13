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
 * HalfEdgeFace.hpp
 *
 *  @date 02.06.2017
 *  @author Lukas Kalbertodt <lukas.kalbertodt@gmail.com>
 */

#ifndef LVR2_GEOMETRY_HALFEDGEFACE_H_
#define LVR2_GEOMETRY_HALFEDGEFACE_H_

#include "BaseMesh.hpp"
#include "Normal.hpp"

namespace lvr2
{

// Forward definitions
template <typename BaseVecT> struct HalfEdge;
template <typename BaseVecT> struct HalfEdgeVertex;
template <typename BaseVecT> struct HalfEdgeMesh;

template <typename BaseVecT>
struct HalfEdgeFace
{
private:
    using Edge = HalfEdge<BaseVecT>;
    using Face = HalfEdgeFace<BaseVecT>;
    using Vertex = HalfEdgeVertex<BaseVecT>;

public:
    HalfEdgeFace(
        EdgeHandle edge,
        Normal<BaseVecT> normal
    )
        : edge(edge), normal(normal) {}

    /// One of the edges bounding this face.
    EdgeHandle edge;

    /// The normal of this face. This can be (and is) calculated from the
    /// face's vertices, but the value is so frequently used, that it's worth
    /// calculating once and putting it here.
    Normal<BaseVecT> normal;
};

} // namespace lvr2

#endif /* LVR2_GEOMETRY_HALFEDGEFACE_H_ */
