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
#include <lvr2/util/StableVector.hpp>
#include <array>

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
    size_t countVertices() final;
    Point<BaseVecT> getPoint(VertexHandle handle) final;
    size_t countFaces() final;
    std::array<Point<BaseVecT>, 3> getPointsOfFace(FaceHandle handle) final;
    std::array<VertexHandle, 3> getVertexHandlesOfFace(FaceHandle handle) final;

    bool debugCheckMeshIntegrity() const;

private:
    StableVector<Edge, EdgeHandle> m_edges;
    StableVector<Face, FaceHandle> m_faces;
    StableVector<Vertex, VertexHandle> m_vertices;

    // ========================================================================
    // = Private helper methods
    // ========================================================================
    Edge& getE(EdgeHandle handle);
    const Edge& getE(EdgeHandle handle) const;
    Face& getF(FaceHandle handle);
    const Face& getF(FaceHandle handle) const;
    Vertex& getV(VertexHandle handle);
    const Vertex& getV(VertexHandle handle) const;

    /**
     * @brief Given two vertices, find the edge pointing from one to the other.
     *
     * @return None, if there exists no such edge.
     */
    OptionalEdgeHandle edgeBetween(VertexHandle fromH, VertexHandle toH);

    EdgeHandle findOrCreateEdgeBetween(VertexHandle fromH, VertexHandle toH);

    /**
     * @brief Adds a new, incomplete edge-pair.
     *
     * This method is private and unsafe, because it leaves some fields
     * uninitialized. The invariants of this mesh are broken after calling this
     * method and the caller has to fix those broken invariants.
     *
     * In particular, no `next` handle is set or changed. The `outgoing` handle
     * of the vertices is not changed (or set) either.
     *
     * @return Both edge handles. The first edge points from v1H to v2H, the
     *         second one points from v2H to v1H.
     */
    pair<EdgeHandle, EdgeHandle> addEdgePair(VertexHandle v1H, VertexHandle v2H);


    /**
     * @brief Iterates over all ingoing edges of one vertex, returning the
     *        first edge that satisfies the given predicate.
     *
     * @return Returns None if `v` does not have an outgoing edge or if no
     *         edge satisfies the predicate.
     */
    template <typename Pred>
    OptionalEdgeHandle findEdgeAroundVertex(VertexHandle vH, Pred pred) const;
};

} // namespace lvr

// #include <lvr2/geometry/HalfEdgeMesh.tcc>
#include <lvr2/geometry/HalfEdgeMesh.tcc>

#endif /* LVR2_GEOMETRY_HALFEDGEMESH_H_ */
