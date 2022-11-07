#pragma once
#ifndef QUERY_CUH
#define QUERY_CUH
#include "lbvh.cuh"
#include "static_priorityqueue.cuh"
#include "vec_math.h"

// default is one nearest neighbor
#ifndef K
#define K 1
#endif

namespace lbvh {

    template<typename Handler>
    __forceinline__ __device__ void handle_node(const BVHNode *node,
                                                        const float3* __restrict__ points,
                                                        const unsigned int* __restrict__ sorted_indices,
                                                        const float3* __restrict__ query_point,
                                                        Handler& handler);
    

    /**
     * Query the bvh using the specified handler
     * @tparam Handler functor template with the following contract:
     *                          struct Handler {
     *                              // the current number of points in the handler (can be 0)
     *                              __device__ unsigned int size() const;
     *
     *                              // the maximum number of points the handler manages (UINT_MAX for infinite
     *                              __device__ unsigned int max_size() const;
     *
     *                              // the maximum distance the handler manages (INFINITY for unbounded)
     *                              __device__ float max_distance() const;
     *
     *                              // add another point to the handler
     *                              __device__ void operator(const float3& point, unsigned int index, float dist);
     *                          }
     */
    template<typename Handler>
    __device__ void query(const BVHNode* __restrict__ nodes,
                             const float3* __restrict__ points,
                             const unsigned int* __restrict__ sorted_indices,
                             unsigned int root_index,
                             const float3* __restrict__ query_point,
                             Handler& handler);
    
}

#endif // QUERY_CUH