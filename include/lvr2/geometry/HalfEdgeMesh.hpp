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
#include <lvr2/attrmaps/StableVector.hpp>
#include <array>
#include <vector>

using std::pair;
using std::array;
using std::vector;

#include "Point.hpp"
#include "Vector.hpp"
#include "BaseMesh.hpp"
#include "HalfEdge.hpp"
#include "HalfEdgeFace.hpp"
#include "HalfEdgeVertex.hpp"

namespace lvr2
{

/**
 * @brief Half-edge data structure implementing the `BaseMesh` interface.
 *
 * This implementation uses a half-edge structure. This encodes many
 * connectivity details explicitly, enabling fast lookup. However, HEMs are
 * primarily intended for non-triangle meshes (variable number of edges per
 * face). Using it for triangle meshes might be overkill and results in a
 * memory overhead.
 */
template<typename BaseVecT>
class HalfEdgeMesh : public BaseMesh<BaseVecT>
{
public:
    using Edge = HalfEdge;
    using Face = HalfEdgeFace;
    using Vertex = HalfEdgeVertex<BaseVecT>;

    // ========================================================================
    // = Implementing the `BaseMesh` interface (see BaseMesh for docs)
    // ========================================================================

    // We declare all metods as `final` to make devirtualization optimizations
    // more likely and effective.
    VertexHandle addVertex(Point<BaseVecT> pos) final;
    FaceHandle addFace(VertexHandle v1H, VertexHandle v2H, VertexHandle v3H) final;
    void removeFace(FaceHandle handle) final;
    EdgeCollapseResult collapseEdge(EdgeHandle edgeH) final;
    void flipEdge(EdgeHandle edgeH) final;

    size_t numVertices() const final;
    size_t numFaces() const final;
    size_t numEdges() const final;

    Point<BaseVecT> getVertexPosition(VertexHandle handle) const final;
    Point<BaseVecT>& getVertexPosition(VertexHandle handle) final;

    array<VertexHandle, 3> getVerticesOfFace(FaceHandle handle) const final;
    array<EdgeHandle, 3> getEdgesOfFace(FaceHandle handle) const final;
    void getNeighboursOfFace(FaceHandle handle, vector<FaceHandle>& facesOut) const final;
    array<VertexHandle, 2> getVerticesOfEdge(EdgeHandle edgeH) const final;
    array<OptionalFaceHandle, 2> getFacesOfEdge(EdgeHandle edgeH) const final;
    void getFacesOfVertex(VertexHandle handle, vector<FaceHandle>& facesOut) const final;
    void getEdgesOfVertex(VertexHandle handle, vector<EdgeHandle>& edgesOut) const final;
    void getNeighboursOfVertex(VertexHandle handle, vector<VertexHandle>& verticesOut) const final;

    // Make sure all default methods from `BaseMesh` are visible
    using BaseMesh<BaseVecT>::getFacesOfVertex;
    using BaseMesh<BaseVecT>::getEdgesOfVertex;
    using BaseMesh<BaseVecT>::getNeighboursOfVertex;


    MeshHandleIteratorPtr<VertexHandle> verticesBegin() const final;
    MeshHandleIteratorPtr<VertexHandle> verticesEnd() const final;
    MeshHandleIteratorPtr<FaceHandle> facesBegin() const final;
    MeshHandleIteratorPtr<FaceHandle> facesEnd() const final;
    MeshHandleIteratorPtr<EdgeHandle> edgesBegin() const final;
    MeshHandleIteratorPtr<EdgeHandle> edgesEnd() const final;


    // ========================================================================
    // = Other public methods
    // ========================================================================

    bool debugCheckMeshIntegrity() const;

private:
    StableVector<HalfEdgeHandle, Edge> m_edges;
    StableVector<FaceHandle, Face> m_faces;
    StableVector<VertexHandle, Vertex> m_vertices;

    // ========================================================================
    // = Private helper methods
    // ========================================================================
    Edge& getE(HalfEdgeHandle handle);
    const Edge& getE(HalfEdgeHandle handle) const;
    Face& getF(FaceHandle handle);
    const Face& getF(FaceHandle handle) const;
    Vertex& getV(VertexHandle handle);
    const Vertex& getV(VertexHandle handle) const;

    /**
     * @brief Converts a half edge handle to a full edge handle
     *
     * @return  The handle with the smaller index of the given half edge and
     *          its twin
     */
    EdgeHandle halfToFullEdgeHandle(HalfEdgeHandle handle) const;

    /**
     * @brief Given two vertices, find the edge pointing from one to the other.
     *
     * @return None, if there exists no such edge.
     */
    OptionalHalfEdgeHandle edgeBetween(VertexHandle fromH, VertexHandle toH);

    /**
     * @brief Attempts to find an edge between the given vertices and, if none
     *        is found, creates a new edge with `addEdgePair()`
     *
     * @return The half edge from `fromH` to `toH`
     */
    HalfEdgeHandle findOrCreateEdgeBetween(VertexHandle fromH, VertexHandle toH);

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
    pair<HalfEdgeHandle, HalfEdgeHandle> addEdgePair(VertexHandle v1H, VertexHandle v2H);


    /**
     * @brief Circulates around the vertex `vH`, calling the `visitor` for each
     *        ingoing edge of the vertex.
     *
     * The edges are visited in clockwise order. The iteration stops if all
     * edges were visited once or if the visitor returns `false`. It has to
     * return `true` to keep circulating. If the vertex has no outgoing edge,
     * this method does nothing.
     */
    template <typename Visitor>
    void circulateAroundVertex(VertexHandle vH, Visitor visitor) const;

    /**
     * @brief Circulates around the vertex `startEdgeH.target`, calling the
     *        `visitor` for each ingoing edge of the vertex.
     *
     * This works exactly as the other overload, but specifically starts at the
     * edge `startEdgeH` instead of `vH.outgoing`.
     */
    template <typename Visitor>
    void circulateAroundVertex(HalfEdgeHandle startEdgeH, Visitor visitor) const;

    /**
     * @brief Iterates over all ingoing edges of one vertex, returning the
     *        first edge that satisfies the given predicate.
     *
     * @return Returns None if `v` does not have an outgoing edge or if no
     *         edge in the circle satisfies the predicate.
     */
    template <typename Pred>
    OptionalHalfEdgeHandle findEdgeAroundVertex(VertexHandle vH, Pred pred) const;

    /**
     * @brief Iterates over all ingoing edges of the vertex `startEdge.target`,
     *        starting at the edge `startEdgeH`, returning the first edge that
     *        satisfies the given predicate.
     *
     * @return Returns None if no edge in the circle satisfies the predicate.
     */
    template <typename Pred>
    OptionalHalfEdgeHandle findEdgeAroundVertex(HalfEdgeHandle startEdgeH, Pred pred) const;

    /**
     * @brief Method for finding contour edges of given face.
     *
     * Returns all edge handles that are contours
     */
    template <typename Pred>
    vector<EdgeHandle> getContourEdgesOfFace(FaceHandle faceH, Pred pred) const;

    /**
     * @brief Get inner edges in counter clockwise order
     */
    array<HalfEdgeHandle, 3> getInnerEdges(FaceHandle handle) const;

    // ========================================================================
    // = Friends
    // ========================================================================
    template<typename> friend class HemEdgeIterator;
};

/// Implementation of the MeshHandleIterator for the HalfEdgeMesh
template<typename HandleT, typename ElemT>
class HemFevIterator : public MeshHandleIterator<HandleT>
{
public:
    HemFevIterator(StableVectorIterator<HandleT, ElemT> iterator) : m_iterator(iterator) {};
    HemFevIterator& operator++();
    bool operator==(const MeshHandleIterator<HandleT>& other) const;
    bool operator!=(const MeshHandleIterator<HandleT>& other) const;
    HandleT operator*() const;

private:
    StableVectorIterator<HandleT, ElemT> m_iterator;
};

template<typename BaseVecT>
class HemEdgeIterator : public MeshHandleIterator<EdgeHandle>
{
public:
    HemEdgeIterator(
        StableVectorIterator<HalfEdgeHandle, HalfEdge> iterator,
        const HalfEdgeMesh<BaseVecT>& mesh
    ) : m_iterator(iterator), m_mesh(mesh) {};

    HemEdgeIterator& operator++();
    bool operator==(const MeshHandleIterator<EdgeHandle>& other) const;
    bool operator!=(const MeshHandleIterator<EdgeHandle>& other) const;
    EdgeHandle operator*() const;

private:
    StableVectorIterator<HalfEdgeHandle, HalfEdge> m_iterator;
    const HalfEdgeMesh<BaseVecT>& m_mesh;
};

} // namespace lvr2

#include <lvr2/geometry/HalfEdgeMesh.tcc>

#endif /* LVR2_GEOMETRY_HALFEDGEMESH_H_ */
