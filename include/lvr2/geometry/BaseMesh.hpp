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

#include "Handles.hpp"
#include "Point.hpp"

namespace lvr2
{

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
};

} // namespace lvr2

#endif /* LVR2_GEOMETRY_BASEMESH_H_ */
