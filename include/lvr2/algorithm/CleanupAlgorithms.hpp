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
 * CleanupAlgorithms.hpp
 */

#ifndef LVR2_ALGORITHM_CLEANUPALGORITHMS_H_
#define LVR2_ALGORITHM_CLEANUPALGORITHMS_H_

#include <lvr2/geometry/BaseMesh.hpp>

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

#include <lvr2/algorithm/CleanupAlgorithms.tcc>

#endif /* LVR2_ALGORITHM_CLEANUPALGORITHMS_H_ */
