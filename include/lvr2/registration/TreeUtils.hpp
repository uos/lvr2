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

/**
 * TreeUtils.hpp
 *
 *  @date May 16, 2019
 *  @author Malte Hillmann
 */
#ifndef TREEUTILS_HPP_
#define TREEUTILS_HPP_

#include "lvr2/types/MatrixTypes.hpp"

namespace lvr2
{

/**
 * @brief Sorts a Point array so that all Points smaller than splitValue are on the left
 *
 * Uses the QuickSort Pivot step
 *
 * @param points     The Point array
 * @param n          The number of Points
 * @param axis       The axis to sort by
 * @param splitValue The value to sort by
 *
 * @return int The number of smaller elements. points + this value gives the start of the greater
 * elements
 */
int splitPoints(Vector3f* points, int n, int axis, double splitValue);

/**
 * @brief Reduces a Point Cloud using an Octree with a minimum Voxel size
 *
 * @param points      The Point Cloud
 * @param n           The number of Points in the Point Cloud
 * @param voxelSize   The minimum size of a Voxel
 * @param maxLeafSize When to stop subdividing Voxels
 *
 * @return int the new number of Points in the Point Cloud
 */
int octreeReduce(Vector3f* points, int n, double voxelSize, int maxLeafSize);

} /* namespace lvr2 */

#endif /* TREEUTILS_HPP_ */
