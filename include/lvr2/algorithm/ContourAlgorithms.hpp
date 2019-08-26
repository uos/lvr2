/**
 * Copyright (c) 2018, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * ContourAlgorithms.hpp
 */

#ifndef LVR2_ALGORITHM_CONTOURALGORITHMS_H_
#define LVR2_ALGORITHM_CONTOURALGORITHMS_H_


#include "lvr2/geometry/BaseMesh.hpp"
#include "lvr2/geometry/Handles.hpp"

#include <vector>

namespace lvr2
{

/**
 * @brief Walks on a boundary contour starting at `startH`.
 *
 * To make the method more generic, it has a `exists` function object which can
 * filter out faces. Each face for which `exists` returns `false` is treated as
 * if it wouldn't exist in the mesh at all! If you don't want to ignore faces,
 * you can just not provide the `exists` parameter to use the other overload.
 *
 * The given `startH` as well as all other edges on the contour mustn't be a
 * lonely edge (when ignoring faces according to `exists`).
 *
 * The edges of the contour are visited in counter-clockwise order. That is, if
 * you look at the contour as if it were a hole in the mesh. If you are
 * thinking of the contour as an outer contour, it's in clockwise order. This
 * matter is pretty confusing and I don't know how to explain it a lot better.
 * Here another attempt:
 *
 * Inner contours are visited in counter-clockwise order and outer ones in
 * clockwise order. Inner contours are commonly referred to as "holes in the
 * mesh", while the outer one is often called "mesh boundary". However, for 3D
 * meshes, the distinction isn't all that clear. What matters here is the
 * planar embedding of the mesh. There are many possible planar embeddings,
 * including the ones that make the "mesh boundary" look like a hole and vice
 * versa.
 *
 * Anyway, we can say this for sure: given one concrete embedding for your
 * mesh, the outer contour's edges are returned in clockwise order and the
 * edges of all inner contours are returned in counter-clockwise order.
 *
 * @param visitor A function object taking two parameters: a `VertexHandle`
 *                and an `EdgeHandle`. The vertex is the vertex of the edge
 *                that comes "before" the edge, speaking about the
 *                direction of visiting the edges.
 * @param exists A function object taking one FaceHandle as parameter and
 *               returning bool. This function decides whether a face should be
 *               treated as existing or not. This is mainly used to walk on the
 *               boundary of clusters. This can be achieved by making `exists`
 *               return `false` for all faces not in the cluster.
 */
template<typename BaseVecT, typename VisitorF, typename PredF>
void walkContour(
    const BaseMesh<BaseVecT>& mesh,
    EdgeHandle startH,
    VisitorF visitor,
    PredF exists
);

/**
 * @brief Like the other overload, but without ignoring any faces.
 */
template<typename BaseVecT, typename VisitorF>
void walkContour(
    const BaseMesh<BaseVecT>& mesh,
    EdgeHandle startH,
    VisitorF visitor
);

/**
 * @brief Walks on a boundary contour starting at `startH` and adds all visited
 *        edges to the given out vector.
 *
 * Uses `walkContour()`. See that function for detailed information.
 */
template<typename BaseVecT, typename PredF>
void calcContourEdges(
    const BaseMesh<BaseVecT>& mesh,
    EdgeHandle startH,
    std::vector<EdgeHandle>& contourOut,
    PredF exists
);

/**
 * @brief Walks on a boundary contour starting at `startH` and adds all visited
 *        edges to the given out vector.
 *
 * Uses `walkContour()`. See that function for detailed information.
 */
template<typename BaseVecT>
void calcContourEdges(
    const BaseMesh<BaseVecT>& mesh,
    EdgeHandle startH,
    std::vector<EdgeHandle>& contourOut
);

/**
 * @brief Walks on a boundary contour starting at `startH` and adds all visited
 *        vertices to the given out vector.
 *
 * Uses `walkContour()`. See that function for detailed information.
 */
template<typename BaseVecT, typename PredF>
void calcContourVertices(
    const BaseMesh<BaseVecT>& mesh,
    EdgeHandle startH,
    std::vector<VertexHandle>& contourOut,
    PredF exists
);

/**
 * @brief Walks on a boundary contour starting at `startH` and adds all visited
 *        vertices to the given out vector.
 *
 * Uses `walkContour()`. See that function for detailed information.
 */
template<typename BaseVecT>
void calcContourVertices(
    const BaseMesh<BaseVecT>& mesh,
    EdgeHandle startH,
    std::vector<VertexHandle>& contourOut
);

} // namespace lvr2

#include "lvr2/algorithm/ContourAlgorithms.tcc"

#endif /* LVR2_ALGORITHM_CONTOURALGORITHMS_H_ */
