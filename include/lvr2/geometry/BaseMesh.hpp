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
#include <vector>
#include <type_traits>
#include <boost/optional.hpp>

using std::vector;
using std::array;
using boost::optional;

#include "Handles.hpp"
#include "Point.hpp"

namespace lvr2
{

/**
 * @brief An iterator for handles in the BaseMesh.
 *
 * Important: This is not a fail fast iterator! If the mesh struct is changed
 * while using an instance of this iterator the behavior is undefined!
 *
 * @tparam HandleT The type of the requested handle
 */
template<typename HandleT>
class MeshHandleIterator
{
    static_assert(std::is_base_of<BaseHandle<Index>, HandleT>::value, "HandleT must inherit from BaseHandle!");
public:
    /// Advances the iterator once. Using the dereference operator afterwards
    /// will yield the next handle.
    virtual MeshHandleIterator& operator++() = 0;
    virtual bool operator==(const MeshHandleIterator& other) const = 0;
    virtual bool operator!=(const MeshHandleIterator& other) const = 0;

    /// Returns the current handle.
    virtual HandleT operator*() const = 0;
};

/// A wrapper for the MeshHandleIterator to save beloved future programmers from dereferencing too much <3
template<typename HandleT>
class MeshHandleIteratorPtr
{
public:
    MeshHandleIteratorPtr(std::unique_ptr<MeshHandleIterator<HandleT>> iter) : m_iter(std::move(iter)) {};
    MeshHandleIteratorPtr& operator++();
    bool operator==(const MeshHandleIteratorPtr& other) const;
    bool operator!=(const MeshHandleIteratorPtr& other) const;
    HandleT operator*() const;
private:
    std::unique_ptr<MeshHandleIterator<HandleT>> m_iter;
};

// Forward declaration
template <typename> class FaceIteratorProxy;
template <typename> class EdgeIteratorProxy;
template <typename> class VertexIteratorProxy;

struct EdgeCollapseResult;
struct EdgeCollapseRemovedFace;

/**
 * @brief Interface for triangle-meshes with adjacency information.
 *
 * This interface represents meshes that contain information about the
 * conectivity of their faces, edges and vertices. They make it possible to
 * access adjacent faces/edges/vertices in constant time.
 *
 * Faces, edges and vertices in these meshes are explicitly represented (the
 * phrase "faces, edge or vertex" is often abbreviated "FEV"). To talk about
 * one specific FEV, so called handles are used. A handle is basically an index
 * which is used to identify a FEV. Note that the internal structures used to
 * represent FEVs are not exposed in this interface. This means you'll never
 * write something like `vertex.outgoingEdge`, but you'll always use methods
 * of this interface to get information about a FEV.
 *
 * Meshes are mainly used to store connectivity information. They are not used
 * to store arbitrary data for each FEV. To do that, you should use FEV maps
 * which allow you to associate arbitrary data with a FEV (and more). For more
 * information about that, please refer to the documentation in `VectorMap`.
 * There is one important exception, though: the 3D position of vertices is
 * stored inside the mesh directly. This is actually rather inconsistent with
 * the whole design, but positions are used a lot -- so it is convenient to
 * store them in the mesh. But this might change in the future.
 *
 * This interface cannot be used for arbitrarily connected meshes. Instead,
 * only manifold meshes can be represented. In particular, this means that each
 * connected component of the mesh has to be a planar graph (you could draw it
 * on a piece of paper without edges crossing). As a consequence we can use
 * terms like "clockwise" and "counter-clockwise" (a property that I think is
 * called "orientable"). When doing that, we assume a planar embedding that
 * shows the face's normals sticking "out of the paper". In easier terms: draw
 * the graph (represented by the mesh) on a paper and  draw it in the way such
 * that you can see the front of all faces. When we talk about "clockwise" and
 * "counter-clockwise" we are talking about this embedding -- when looking at
 * the face.
 */
template<typename BaseVecT>
class BaseMesh
{
public:
    virtual ~BaseMesh() {}

    // =======================================================================
    // Pure virtual methods (need to be implemented)
    // =======================================================================

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
     * @brief Removes the given face and all (if not connected to any other face/edge/vertex) connected edges and
     *        vertices.
     */
    virtual void removeFace(FaceHandle handle) = 0;

    /**
     * @brief Merges the two vertices connected by the given edge.
     *
     * If existing, the two neighboring faces or triangles without faces and
     * their edges are removed and replaced by two new edges
     * The vertices at the start and end of the given edge are removed and
     * replaced by a new vertex at the center of the previous vertex positions
     *
     * @return  An EdgeCollapseResult that contains handles of the removed
     *          faces and edges and the new vertex and the new edges
     */
    virtual EdgeCollapseResult collapseEdge(EdgeHandle edgeH) = 0;

    /**
     * @brief Returns the number of vertices in the mesh.
     */
    virtual size_t numVertices() const = 0;

    /**
     * @brief Returns the number of faces in the mesh.
     */
    virtual size_t numFaces() const = 0;

    /**
     * @brief Returns the number of edges in the mesh.
     */
    virtual size_t numEdges() const = 0;

    /**
     * @brief Get the position of the given vertex.
     */
    virtual Point<BaseVecT> getVertexPosition(VertexHandle handle) const = 0;

    /**
     * @brief Get a ref to the position of the given vertex.
     */
    virtual Point<BaseVecT>& getVertexPosition(VertexHandle handle) = 0;

    /**
     * @brief Get the three vertices surrounding the given face.
     *
     * @return The vertex-handles in counter-clockwise order.
     */
    virtual array<VertexHandle, 3> getVerticesOfFace(FaceHandle handle) const = 0;

    /**
     * @brief Get the three edges surrounding the given face.
     *
     * @return The edge-handles in counter-clockwise order.
     */
    virtual array<EdgeHandle, 3> getEdgesOfFace(FaceHandle handle) const = 0;

    /**
     * @brief Get face handles of the neighbours of the requested face.
     *
     * The face handles are written into the `facesOut` vector. This is done
     * to reduce the number of heap allocations if this method is called in
     * a loop. If you are not calling it in a loop or can't, for some reason,
     * take advantages of this method's signature, you can call the other
     * overload of this method which just returns the vector. Such convinient.
     *
     * Note: you probably should remember to `clear()` the vector before
     * passing it into this method.
     *
     * @param facesOut The face-handles of the neighbours of `handle` will be
     *                 written into this vector in counter-clockwise order.
     *                 There are at most three neighbours of a face, so this
     *                 method will push 0, 1, 2 or 3 handles to `facesOut`.
     */
    virtual void getNeighboursOfFace(FaceHandle handle, vector<FaceHandle>& facesOut) const = 0;

    /**
     * @brief Get the two vertices of an edge.
     *
     * The order of the vertices is not specified
     */
    virtual array<VertexHandle, 2> getVerticesOfEdge(EdgeHandle edgeH) const = 0;

    /**
     * @brief Get the two faces of an edge.
     *
     * The order of the faces is not specified
     */
    virtual array<OptionalFaceHandle, 2> getFacesOfEdge(EdgeHandle edgeH) const = 0;

    /**
     * @brief Get a list of faces the given vertex belongs to.
     *
     * The face handles are written into the `facesOut` vector. This is done
     * to reduce the number of heap allocations if this method is called in
     * a loop. If you are not calling it in a loop or can't, for some reason,
     * take advantages of this method's signature, you can call the other
     * overload of this method which just returns the vector. Such convinient.
     *
     * Note: you probably should remember to `clear()` the vector before
     * passing it into this method.
     *
     * @param facesOut The handles of the faces around `handle` will be written
     *                 into this vector in clockwise order.
     */
    virtual void getFacesOfVertex(VertexHandle handle, vector<FaceHandle>& facesOut) const = 0;

    /**
     * @brief Get a list of edges around the given vertex.
     *
     * The face handles are written into the `edgesOut` vector. This is done
     * to reduce the number of heap allocations if this method is called in
     * a loop. If you are not calling it in a loop or can't, for some reason,
     * take advantages of this method's signature, you can call the other
     * overload of this method which just returns the vector. Such convinient.
     *
     * Note: you probably should remember to `clear()` the vector before
     * passing it into this method.
     *
     * @param edgesOut The handles of the edges around `handle` will be written
     *                 into this vector in clockwise order.
     */
    virtual void getEdgesOfVertex(VertexHandle handle, vector<EdgeHandle>& edgesOut) const = 0;

    /**
     * @brief Get vertex handles of the neighbours of the requested vertex.
     *
     * The vertex handles are written into the `verticesOut` vector. This is
     * done to reduce the number of heap allocations if this method is called
     * in a loop. If you are not calling it in a loop or can't, for some
     * reason, take advantages of this method's signature, you can call the
     * other overload of this method which just returns the vector. Such
     * convinient.
     *
     * Note: you probably should remember to `clear()` the vector before
     * passing it into this method.
     *
     * @param verticesOut The vertex-handles of the neighbours of `handle` will
     *                    be written into this vector in clockwise order.
     */
    virtual void getNeighboursOfVertex(VertexHandle handle, vector<VertexHandle>& verticesOut) const = 0;


    /**
     * @brief Returns an iterator to the first vertex of this mesh.
     *
     * @return When dereferenced, this iterator returns a handle to the current vertex.
     */
    virtual MeshHandleIteratorPtr<VertexHandle> verticesBegin() const = 0;

    /**
     * @brief Returns an iterator to the element following the last vertex of this mesh.
     */
    virtual MeshHandleIteratorPtr<VertexHandle> verticesEnd() const = 0;

    /**
     * @brief Returns an iterator to the first face of this mesh.
     *
     * @return When dereferenced, this iterator returns a handle to the current face.
     */
    virtual MeshHandleIteratorPtr<FaceHandle> facesBegin() const = 0;

    /**
     * @brief Returns an iterator to the element following the last face of this mesh.
     */
    virtual MeshHandleIteratorPtr<FaceHandle> facesEnd() const = 0;

    /**
     * @brief Returns an iterator to the first edge of this mesh.
     *
     * @return When dereferenced, this iterator returns a handle to the current edge.
     */
    virtual MeshHandleIteratorPtr<EdgeHandle> edgesBegin() const = 0;

    /**
     * @brief Returns an iterator to the element following the last edge of this mesh.
     */
    virtual MeshHandleIteratorPtr<EdgeHandle> edgesEnd() const = 0;

    // =======================================================================
    // Provided methods (already implemented)
    // =======================================================================

    /**
     * @brief Get the points of the requested face.
     *
     * @return The points of the vertices in counter-clockwise order.
     */
    virtual array<Point<BaseVecT>, 3> getVertexPositionsOfFace(FaceHandle handle) const;

    /**
     * @brief Calc and return the centroid of the requested face.
     */
    Point<BaseVecT> calcFaceCentroid(FaceHandle handle) const;

    /**
     * @brief Calc and return the area of the requested face.
     */
    typename BaseVecT::CoordType calcFaceArea(FaceHandle handle) const;

    /**
     * @brief Get face handles of the neighbours of the requested face.
     *
     * This method is implemented using the pure virtual method
     * `getNeighboursOfFace(FaceHandle, vector<FaceHandle>&)`. If you are
     * calling this method in a loop, you should probably call the more manual
     * method (with the out vector) to avoid useless heap allocations.
     *
     * @return The face-handles of the neighbours in counter-clockwise order.
     */
    virtual vector<FaceHandle> getNeighboursOfFace(FaceHandle handle) const;

    /**
     * @brief Get a list of faces the given vertex belongs to.
     *
     * This method is implemented using the pure virtual method
     * `getFacesOfVertex(VertexHandle, vector<FaceHandle>&)`. If you are
     * calling this method in a loop, you should probably call the more manual
     * method (with the out vector) to avoid useless heap allocations.
     *
     * @return The face-handles in counter-clockwise order.
     */
    virtual vector<FaceHandle> getFacesOfVertex(VertexHandle handle) const;

    /**
     * @brief Get a list of edges around the given vertex.
     *
     * This method is implemented using the pure virtual method
     * `getEdgesOfVertex(VertexHandle, vector<EdgeHandle>&)`. If you are
     * calling this method in a loop, you should probably call the more manual
     * method (with the out vector) to avoid useless heap allocations.
     *
     * @return The edge-handles in counter-clockwise order.
     */
    virtual vector<EdgeHandle> getEdgesOfVertex(VertexHandle handle) const;

    /**
     * @brief Get a list of vertices around the given vertex.
     *
     * This method is implemented using the pure virtual method
     * `getNeighboursOfVertex(VertexHandle, vector<EdgeHandle>&)`. If you are
     * calling this method in a loop, you should probably call the more manual
     * method (with the out vector) to avoid useless heap allocations.
     *
     * @return The vertex-handles in clockwise order.
     */
    virtual vector<VertexHandle> getNeighboursOfVertex(VertexHandle handle) const;

    /**
     * @brief Method for usage in range-based for-loops.
     *
     * Returns a simple proxy object that uses `facesBegin()` and `facesEnd()`.
     */
    virtual FaceIteratorProxy<BaseVecT> faces() const;

    /**
     * @brief Method for usage in range-based for-loops.
     *
     * Returns a simple proxy object that uses `edgesBegin()` and `edgesEnd()`.
     */
    virtual EdgeIteratorProxy<BaseVecT> edges() const;

    /**
     * @brief Method for usage in range-based for-loops.
     *
     * Returns a simple proxy object that uses `verticesBegin()` and `verticesEnd()`.
     */
    virtual VertexIteratorProxy<BaseVecT> vertices() const;
};

template <typename BaseVecT>
class FaceIteratorProxy
{
public:
    MeshHandleIteratorPtr<FaceHandle> begin() const;
    MeshHandleIteratorPtr<FaceHandle> end() const;

private:
    FaceIteratorProxy(const BaseMesh<BaseVecT>& mesh) : m_mesh(mesh) {}
    const BaseMesh<BaseVecT>& m_mesh;
    friend BaseMesh<BaseVecT>;
};

template <typename BaseVecT>
class EdgeIteratorProxy
{
public:
    MeshHandleIteratorPtr<EdgeHandle> begin() const;
    MeshHandleIteratorPtr<EdgeHandle> end() const;

private:
    EdgeIteratorProxy(const BaseMesh<BaseVecT>& mesh) : m_mesh(mesh) {}
    const BaseMesh<BaseVecT>& m_mesh;
    friend BaseMesh<BaseVecT>;
};

template <typename BaseVecT>
class VertexIteratorProxy
{
public:
    MeshHandleIteratorPtr<VertexHandle> begin() const;
    MeshHandleIteratorPtr<VertexHandle> end() const;

private:
    VertexIteratorProxy(const BaseMesh<BaseVecT>& mesh) : m_mesh(mesh) {}
    const BaseMesh<BaseVecT>& m_mesh;
    friend BaseMesh<BaseVecT>;
};

struct EdgeCollapseRemovedFace
{
    /// A face adjacent to the collapsed edge which was removed
    OptionalFaceHandle removedFace;

    /// The edges of the removed face (excluding the collapsed edge itself)
    array<EdgeHandle, 2> removedEdges;

    /// The edge that was inserted to replace the removed face
    EdgeHandle newEdge;

    EdgeCollapseRemovedFace(
        OptionalFaceHandle removedFace,
        array<EdgeHandle, 2> removedEdges,
        EdgeHandle newEdge
    ) : removedFace(removedFace), removedEdges(removedEdges), newEdge(newEdge) {};
};

struct EdgeCollapseResult
{
    /// The vertex which was inserted to replace the collapsed edge
    VertexHandle midPoint;

    /// The (face) neighbors of the edge which might have been removed. If so,
    /// the entry is not `none` and contains information about the invalidated
    /// handles and the replacement edge.
    array<optional<EdgeCollapseRemovedFace>, 2> neighbors;

    EdgeCollapseResult(VertexHandle midPoint) : midPoint(midPoint) {};
};

} // namespace lvr2

#include <lvr2/geometry/BaseMesh.tcc>

#endif /* LVR2_GEOMETRY_BASEMESH_H_ */
