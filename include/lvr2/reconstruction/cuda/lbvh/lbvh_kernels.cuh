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
#ifndef LVR2_LBVH_KERNELS_CUH
#define LVR2_LBVH_KERNELS_CUH

#define HASH_64 1 // use 64 bit morton codes

#include <cuda_runtime.h>

#include "aabb.cuh"
#include "lbvh.cuh"
#include "morton_code.cuh"

namespace lvr2
{

namespace lbvh 
{
/**
 * @brief   Cuda kernel that calculates morton codes 
 *          of axis aligned bounding boxes
 * 
 * @param aabbs the bounding boxes
 * @param extent the extent of the point cloud
 * @param morton_codes Stores the morton codes
 * @param N number of aabbs
 */
__global__ 
void compute_morton_kernel(
    const AABB* __restrict__ aabbs, 
    const AABB* __restrict__ extent, 
    unsigned long long int* morton_codes, 
    unsigned int N
); 

/**
 * @brief   Cuda kernel that calculates morton codes 
 *          of points
 * 
 * @param points the points
 * @param extent the extent of the point cloud
 * @param morton_codes Stores the morton codes
 * @param N number of points
 */
__global__ 
void compute_morton_points_kernel(
    float* __restrict__ const points,  
    AABB* __restrict__ const extent,
    unsigned long long int* morton_codes,
    unsigned int N
);

/**
 * @brief A Cuda kernel that initializes the tree
 * 
 * @param nodes nodes of the tree
 * @param sorted_aabbs sorted bounding boxes
 * @param n number of leaf nodes
 */  
__global__ 
void initialize_tree_kernel(
    BVHNode *nodes,
    const AABB *sorted_aabbs,
    unsigned int N
);

/**
 * @brief A cuda kernel that constructs the tree
 * 
 * @param nodes nodes of the tree
 * @param root_index root index of the tree
 * @param sorted_morton_codes sorted morton codes of the points
 * @param N number of leaf nodes
 */
__global__ 
void construct_tree_kernel(
    BVHNode *nodes,
    unsigned int* root_index,
    const unsigned long long int *sorted_morton_codes,
    unsigned int N
);

/**
 * @brief A cuda kernel that optimizes the tree
 * 
 * @param nodes nodes of the tree
 * @param root_index root index of the tree
 * @param valid stores which node is still valid after optimization
 * @param max_node_size the maximum leaf size
 * @param N number of leaf nodes
 */
__global__ 
void optimize_tree_kernel(
    BVHNode *nodes,
    unsigned int* root_index,
    unsigned int* valid,
    unsigned int max_node_size,
    unsigned int N
);

/**
 * @brief   A cuda kernel which computes the free indices after 
 *          the tree optimization
 * 
 * @param valid_sums prefix sums of the valid array
 * @param isum the isum array
 * @param free_indices stores the free indices
 * @param N number of valid nodes after optimization
 */
__global__ 
void compute_free_indices_kernel(
    const unsigned int* valid_sums,
    const unsigned int* isums,
    unsigned int* free_indices,
    unsigned int N
);

/**
 * @brief A cuda kernel which compacts the tree after optimization
 * 
 * @param nodes the nodes of the tree
 * @param root_index root index of the tree
 * @param valid_sums prefix sums of the valid array
 * @param free_positions free node positions atfer optimization
 * @param first_moved index of the first node that will be moved
 * @param node_cnt_new number of valid nodes after optimization
 * @param N number of nodes
 */
__global__ 
void compact_tree_kernel(
    BVHNode *nodes,
    unsigned int* root_index,
    const unsigned int* valid_sums,
    const unsigned int* free_positions,
    unsigned int first_moved,
    unsigned int node_cnt_new,
    unsigned int N
);

} // namespace lbvh

} // namespace lvr2


#endif // LVR2_LBVH_KERNELS_CUH