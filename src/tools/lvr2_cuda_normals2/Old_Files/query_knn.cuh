#pragma once
#ifndef QUERY_KNN_CUH
#define QUERY_KNN_CUH

#include "query.cuh"
#include "static_priorityqueue.cuh"

// default is one nearest neighbor
#ifndef K
#define K 1
#endif

namespace lbvh {
    __device__ void query_knn(const BVHNode* __restrict__ nodes,
                             const float3* __restrict__ points,
                             const unsigned int* __restrict__ sorted_indices,
                             unsigned int root_index,
                             const float3* __restrict__ query_point,
                             StaticPriorityQueue<float, K>& queue);
   

    __device__ StaticPriorityQueue<float, K> query_knn(const BVHNode* __restrict__ nodes,
                             const float3* __restrict__ points,
                             const unsigned int* __restrict__ sorted_indices,
                             unsigned int root_index,
                             const float3* __restrict__ query_point,
                             const float max_radius);
}

#endif // QUERY_KNN_CUH