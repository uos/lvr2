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
 * BaseMesh.hpp
 *
 *  @date 02.06.2017
 *  @author Lukas Kalbertodt <lukas.kalbertodt@gmail.com>
 */

#ifndef LVR2_GEOMETRY_BASEMESH_H_
#define LVR2_GEOMETRY_BASEMESH_H_

#include <cstdint>
#include <array>
#include <type_traits>

#include "Handles.hpp"
#include "Point.hpp"

namespace lvr2
{

/**
 * @brief An iterator for handles in the BaseMesh.
 *
 * Important: This is not a fail fast iterator! If the mesh is changed while using an instance of this
 * iterator the behavior is undefined!
 *
 * @tparam HandleT The type of the requested handle
 */
template<typename HandleT>
class MeshHandleIterator
{
    static_assert(std::is_base_of<BaseHandle<Index>, HandleT>::value, "HandleT must inherit from BaseHandle!");
public:
    /// Iterates to next vertex in the mesh
    virtual MeshHandleIterator& operator++(int) = 0;
    virtual bool operator==(const MeshHandleIterator& other) const = 0;
    virtual bool operator!=(const MeshHandleIterator& other) const = 0;

    /// Returns the handle of the current vertex
    virtual HandleT operator*() const = 0;
};

/// A wrapper for the MeshHandleIterator to save beloved future programmers from dereferencing too much <3
template<typename HandleT>
class MeshHandleIteratorPtr
{
public:
    MeshHandleIteratorPtr(std::unique_ptr<MeshHandleIterator<HandleT>> iter) : m_iter(std::move(iter)) {};
    MeshHandleIteratorPtr& operator++(int);
    bool operator==(const MeshHandleIteratorPtr& other) const;
    bool operator!=(const MeshHandleIteratorPtr& other) const;
    HandleT operator*() const;
private:
    std::unique_ptr<MeshHandleIterator<HandleT>> m_iter;
};

/**
 * @brief Interface for triangle-meshes with information about face neighborhood.
 *
 * This interface represents meshes that contain information about the
 * conectivity of their faces, edges and vertices. They make it possible to
 * access adjacent faces/edges/vertices in constant time.
 *
 * TODO: extend this documentation once the interface is more fleshed out!
 */
template<typename BaseVecT>
class BaseMesh
{
public:
    virtual ~BaseMesh() {}

    /**
     * @brief Adds a vertex with the given position to the mesh.
     *
     * The vertex is not connected to anything after calling this method. To
     * add this vertex to a face, use `addFace()`.
     *
     * @return A handle to access the inserted vertex later.
     */
    virtual VertexHandle addVertex(Point<BaseVecT> pos) = 0;

    /**
     * @brief Creates a face connecting the three given vertices.
     *
     * Important: The face's vertices have to be given in front-face counter-
     * clockwise order. This means that, when looking at the face's front, the
     * vertices would appear in counter-clockwise order. Or in more mathy
     * terms: the face's normal is equal to (v1 - v2) x (v1 - v3) in the
     * right-handed coordinate system (where `x` is cross-product).
     *
     * @return A handle to access the inserted face later.
     */
    virtual FaceHandle addFace(VertexHandle v1, VertexHandle v2, VertexHandle v3) = 0;

    /**
     * @brief Return the number of vertices in the mesh.
     */
    virtual size_t numVertices() const = 0;

    /**
     * @brief Return the number of faces in the mesh.
     */
    virtual size_t numFaces() const = 0;

    /**
     * @brief Get the point of the requested vertex.
     */
    virtual Point<BaseVecT> getPoint(VertexHandle handle) const = 0;

    /**
     * @brief Get the points of the requested face.
     *
     * @return The points of the vertices in counter-clockwise order.
     */
    virtual std::array<Point<BaseVecT>, 3> getPointsOfFace(FaceHandle handle) const = 0;

    /**
     * @brief Get vertex handles of the requested face.
     *
     * @return The vertex-handles in counter-clockwise order.
     */
    virtual std::array<VertexHandle, 3> getVertexHandlesOfFace(FaceHandle handle) const = 0;

    /**
     * @brief Create an iterator at the start of the vertices of the current mesh
     * @return When dereferenced, this iterator returns a handle to the current vertex
     */
    virtual MeshHandleIteratorPtr<VertexHandle> verticesBegin() const = 0;

    /**
     * @brief Create an iterator at the end of the vertices of the current mesh
     * @return When dereferenced, this iterator returns a handle to the current vertex
     */
    virtual MeshHandleIteratorPtr<VertexHandle> verticesEnd() const = 0;

    /**
     * @brief Create an iterator at the start of the faces of the current mesh
     * @return When dereferenced, this iterator returns a handle to the current face
     */
    virtual MeshHandleIteratorPtr<FaceHandle> facesBegin() const = 0;

    /**
     * @brief Create an iterator at the end of the faces of the current mesh
     * @return When dereferenced, this iterator returns a handle to the current face
     */
    virtual MeshHandleIteratorPtr<FaceHandle> facesEnd() const = 0;

    /**
     * @brief Create an iterator at the start of the edges of the current mesh
     * @return When dereferenced, this iterator returns a handle to the current edge
     */
    virtual MeshHandleIteratorPtr<EdgeHandle> edgesBegin() const = 0;

    /**
     * @brief Create an iterator at the end of the edges of the current mesh
     * @return When dereferenced, this iterator returns a handle to the current edge
     */
    virtual MeshHandleIteratorPtr<EdgeHandle> edgesEnd() const = 0;
};

} // namespace lvr2

#include <lvr2/geometry/BaseMesh.tcc>

#endif /* LVR2_GEOMETRY_BASEMESH_H_ */
