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
#include <utility>
#include <vector>

using std::vector;
using std::pair;

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
    FaceHandle addFace(VertexHandle v1, VertexHandle v2, VertexHandle v3) final;

private:
    vector<Edge> m_edges;
    vector<Face> m_faces;
    vector<Vertex> m_vertices;

    // ========================================================================
    // = Private helper methods
    // ========================================================================
    Edge& getE(EdgeHandle handle);
    Face& getF(FaceHandle handle);
    Vertex& getV(VertexHandle handle);

    /**
     * @brief Given two vertices, find the edge pointing from one to the other.
     *
     * @return None, if there exists no such edge.
     */
    OptionalEdgeHandle edgeBetween(VertexHandle fromH, VertexHandle toH);

    /**
     * @brief Adds a new edge-pair with invalid `next` and `target` fields.
     *
     * @return The handles of both inserted edges. Those edges are not
     *         connected to a vertex yet and have an invalid `next` handle.
     *         Every calling method need to fix those two fields to avoid
     *         a corrupted mesh.
     */
    pair<EdgeHandle, EdgeHandle> addEdgePair();
};

} // namespace lvr

// #include <lvr2/geometry/HalfEdgeMesh.tcc>
#include <lvr2/geometry/HalfEdgeMesh.tcc>

#endif /* LVR2_GEOMETRY_HALFEDGEMESH_H_ */
