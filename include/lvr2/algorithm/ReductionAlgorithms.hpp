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
 * ReductionAlgorithms.hpp
 *
 * All these algorithms are based on/inspired by:
 *
 * Melax, Stan. "A simple, fast, and effective polygon reduction algorithm."
 * Game Developer 11 (1998): 44-49.
 */

#ifndef LVR2_ALGORITHM_REDUCTIONALGORITHMS_H_
#define LVR2_ALGORITHM_REDUCTIONALGORITHMS_H_

#include <boost/optional.hpp>



namespace lvr2
{

/**
 * @brief Collapses `count` many edges of `mesh` for which `collapseCost`
 *        returns the smallest values.
 *
 * This algorithm collapses edges in the given mesh. The order of edge
 * collapses is determined by `collapseCost`: the edge with the smallest cost
 * is collapsed first, if that is possible. In case the edge is not
 * collapsable, it is ignored and the next edge is choosen. After each
 * collapse, the cost function is called again to update the costs for all
 * edges that could have been affected, which are the edges of all faces which
 * touch the vertex the edge was collapsed to.
 *
 * This algorithm stops when either `count` many edges have been collapsed or
 * if there are no collapsable edges left.
 *
 * @param[in] count Number of edges to collapse
 * @param[in, out] faceNormals A face map storing valid normals of all faces in
 *                             the mesh. This map is altered by this algorithm
 *                             according to the changes done in the mesh.
 * @param[in] collapseCost Function which is called with an edge handle and a
 *                         FaceMap containing normals; it is expected to return
 *                         an optional float. `boost::none` means that this
 *                         edge cannot be collapsed.
 *
 * @return The number of edges actually collapsed.
 */
template<typename BaseVecT, typename CostF>
size_t iterativeEdgeCollapse(
    BaseMesh<BaseVecT>& mesh,
    const size_t count,
    FaceMap<Normal<typename BaseVecT::CoordType>>& faceNormals,
    CostF collapseCost
);

/**
 * @brief Like `iterativeEdgeCollapse` but with a fixed cost function.
 */
template<typename BaseVecT>
size_t simpleMeshReduction(
    BaseMesh<BaseVecT>& mesh,
    const size_t count,
    FaceMap<Normal<typename BaseVecT::CoordType>>& faceNormals
);

} // namespace lvr2

#include "lvr2/algorithm/ReductionAlgorithms.tcc"

#endif /* LVR2_ALGORITHM_REDUCTIONALGORITHMS_H_ */
