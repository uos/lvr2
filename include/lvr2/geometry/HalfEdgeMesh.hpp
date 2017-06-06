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
 * HalfEdgeMesh.hpp
 *
 *  @date 02.06.2017
 *  @author Lukas Kalbertodt <lukas.kalbertodt@gmail.com>
 */

#ifndef LVR2_GEOMETRY_HALFEDGEMESH_H_
#define LVR2_GEOMETRY_HALFEDGEMESH_H_

#include <cstdint>
#include <vector>

using std::vector;

#include "BaseMesh.hpp"
#include "HalfEdge.hpp"
#include "HalfEdgeFace.hpp"
#include "HalfEdgeVertex.hpp"

namespace lvr2
{

/**
 * @brief
 */
template<typename BaseVecT>
class HalfEdgeMesh : public BaseMesh<BaseVecT>
{
public:
    using Edge = HalfEdge<BaseVecT>;
    using Face = HalfEdgeFace<BaseVecT>;
    using Vertex = HalfEdgeVertex<BaseVecT>;

    using Index = typename BaseMesh<BaseVecT>::Index;

    using EdgeHandle = typename BaseMesh<BaseVecT>::EdgeHandle;
    using FaceHandle = typename BaseMesh<BaseVecT>::FaceHandle;
    using VertexHandle = typename BaseMesh<BaseVecT>::VertexHandle;

    using OptionalEdgeHandle = typename BaseMesh<BaseVecT>::OptionalEdgeHandle;
    using OptionalFaceHandle = typename BaseMesh<BaseVecT>::OptionalFaceHandle;
    using OptionalVertexHandle = typename BaseMesh<BaseVecT>::OptionalVertexHandle;


    VertexHandle addVertex(Point<BaseVecT> pos) final;

private:
    vector<Edge> m_edges;
    vector<Face> m_faces;
    vector<Vertex> m_vertices;
};

} // namespace lvr


#include "HalfEdgeMesh.tcc"

#endif /* LVR2_GEOMETRY_HALFEDGEMESH_H_ */
