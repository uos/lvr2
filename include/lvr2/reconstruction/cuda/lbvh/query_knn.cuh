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
#include "query.cuh"
#include "static_priorityqueue.cuh"

// default is one nearest neighbor
#ifndef K
#define K 1
#endif

namespace lvr2
{

namespace lbvh 
{
    __forceinline__ __device__ void query_knn(
        const BVHNode* __restrict__ nodes,
        const float* __restrict__ points,          // Changed from float3* to float*
        const unsigned int* __restrict__ sorted_indices,
        unsigned int root_index,
        const float3* __restrict__ query_point,
        StaticPriorityQueue<float, K>& queue)
    {
        query<StaticPriorityQueue<float, K>>(nodes, points, sorted_indices, root_index, query_point, queue);
    }
    
    __forceinline__ __device__ StaticPriorityQueue<float, K> query_knn(
        const BVHNode* __restrict__ nodes,
        const float* __restrict__ points,          // Changed from float3* to float*
        const unsigned int* __restrict__ sorted_indices,
        unsigned int root_index,
        const float3* __restrict__ query_point,
        const float max_radius
    )
    {
        StaticPriorityQueue<float, K> queue(max_radius);
        query_knn(nodes, points, sorted_indices, root_index, query_point, queue);
        return queue;
    }
} // namespace lbvh

} // namespace lvr2

