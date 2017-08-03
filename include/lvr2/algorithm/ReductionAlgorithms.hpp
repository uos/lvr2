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
 * ReductionAlgorithms.hpp
 */

#ifndef LVR2_ALGORITHM_REDUCTIONALGORITHMS_H_
#define LVR2_ALGORITHM_REDUCTIONALGORITHMS_H_

namespace lvr2
{

/**
 * @brief Collapses `count` many edges of `mesh` for which `collapseCost`
 *        returns the smallest values.
 *
 * This algorithm collapses edges in the given mesh. The order of edge
 * collapses is determined by `collapseCost`: the edge with the smallest cost
 * is collapsed first. After each collapse, the cost function is called again
 * to update the costs for all edges that could have been affected, which are
 * the edges of all faces which touch the vertex the edge was collapsed to.
 *
 * This algorithm stops when either `count` many edges have been collapsed or
 * if the mesh has no edges left.
 *
 * @param count Number of edges to collapse
 * @param collapseCost Function which gets an edge handle and returns a float
 */
template<typename BaseVecT, typename CostF>
void iterativeEdgeCollapse(BaseMesh<BaseVecT>& mesh, const size_t count, CostF collapseCost);


template<typename BaseVecT>
float collapseCostSimpleNormalDiff(
    const BaseMesh<BaseVecT>& mesh,
    const FaceMap<Normal<BaseVecT>>& normals,
    EdgeHandle eH
);

} // namespace lvr2

#include <lvr2/algorithm/ReductionAlgorithms.tcc>

#endif /* LVR2_ALGORITHM_REDUCTIONALGORITHMS_H_ */
