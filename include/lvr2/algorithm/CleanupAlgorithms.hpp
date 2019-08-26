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
 * CleanupAlgorithms.hpp
 */

#ifndef LVR2_ALGORITHM_CLEANUPALGORITHMS_H_
#define LVR2_ALGORITHM_CLEANUPALGORITHMS_H_

#include "lvr2/geometry/BaseMesh.hpp"

namespace lvr2
{

/**
 * @brief Removes faces with a high number of boundary edges.
 *
 * Faces which have 2 or 3 adjacent boundary edges, are removed. If the face
 * is adjacent to only one boundary edge, it is deleted if the face's area is
 * smaller than `areaThreshold`.
 */
template<typename BaseVecT>
void cleanContours(BaseMesh<BaseVecT>& mesh, int iterations, float areaThreshold);


/**
 * @brief Fills holes consisting of less than or equal to `maxSize` edges.
 *
 * It is a rather simple algorithm, really. For each connected part of the mesh
 * it assumes that each boundary-contour except the one with most edges is a
 * hole. These holes are then filled by collapsing all of their boundary edges
 * until no collapsable edge is left (which happens in any case when the hole
 * has only three edges left). If the remaining hole has only three edges after
 * the previous step, it is filled by simply inserting a triangle.
 *
 * Important: this algorithm assumes that the mesh doesn't contain any lonely
 * edges.
 *
 * @return The number of holes that this algorithm wasn't able to fill.
 */
template<typename BaseVecT>
size_t naiveFillSmallHoles(BaseMesh<BaseVecT>& mesh, size_t maxSize, bool collapseOnly);

} // namespace lvr2

#include "lvr2/algorithm/CleanupAlgorithms.tcc"

#endif /* LVR2_ALGORITHM_CLEANUPALGORITHMS_H_ */
