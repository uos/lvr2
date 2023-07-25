/**
 * Copyright (c) 2023, University Osnabrück
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

#pragma once
#ifndef LVR2_LBVH_NORMALS_KERNEL
#define LVR2_LBVH_NORMALS_KERNEL

namespace lvr2
{

namespace lbvh 
{

/**
 * @brief   A Cuda kernel that calculates surface normals of points
 *          with given kNNs 
 * 
 * @param points            Points of the dataset
 * @param queries           Query points for which the normals are calculated
 * @param num_queries       Number of queries
 * @param K                 Max number of neighbors for each point
 * @param n_neighbors_in    Number of neighbors for each point
 * @param indices_in        Indices of the neighbors
 * @param normals           Stores the calculated normals
 * @param flip_x            x coordinate that the normals will be flipped to
 * @param flip_y            y coordinate that the normals will be flipped to
 * @param flip_z            z coordinate that the normals will be flipped to
 */
__global__
void calculate_normals_kernel
    (float* points,
    float* queries, 
    size_t num_queries, 
    int K,
    unsigned int* n_neighbors_in, 
    unsigned int* indices_in, 
    float* normals,
    float flip_x, 
    float flip_y, 
    float flip_z
);

} // namespace lbvh

} // namespace lvr2

#endif // LVR2_LBVH_NORMALS_KERNEL