#pragma once
#ifndef LVR2_KNN_NORMALS_KERNEL_CUH
#define LVR2_KNN_NORMALS_KERNEL_CUH

#include "lbvh.cuh"

namespace lbvh {

__global__ void knn_normals_kernel(
    const BVHNode *nodes,
    const float* __restrict__ points,         
    const unsigned int* __restrict__ sorted_indices,
    const unsigned int root_index,
    const float max_radius,
    const float* __restrict__ query_points,    
    const unsigned int* __restrict__ sorted_queries,
    const unsigned int num_queries, 
    // custom parameters
    float* normals
    //float flip_x=1000000.0, float flip_y=1000000.0, float flip_z=1000000.0
);
} // namespace lbvh

#endif // LVR2_KNN_NORMALS_KERNEL_CUH