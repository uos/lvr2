#pragma once
#ifndef LVR2_KNN_NORMALS_KERNEL_CUH
#define LVR2_KNN_NORMALS_KERNEL_CUH

namespace lvr2
{
namespace lbvh
{

/**
 * @brief   A Cuda kernel that performs a kNN search on the LBVH and
 *          calculates the surface normals by doing an approximated
 *          iterative PCA
 * 
 * @param nodes             Nodes of the LBVH
 * @param points            Points of the dataset
 * @param sorted_indices    Sorted indices of the points
 * @param root_index        Index of the LBVH root node
 * @param max_radius        Maximum radius for radius search
 * @param query_points      Query points for which the normals are calculated
 * @param sorted_queries    Sorted indices of the query points
 * @param num_queries       Number of queries
 * @param normals           Stores the calculated normals
 */
extern "C" __global__ void knn_normals_kernel(
    const BVHNode *nodes,
    const float* __restrict__ points,         
    const unsigned int* __restrict__ sorted_indices,
    const unsigned int root_index,
    const float max_radius,
    const float* __restrict__ query_points,    
    const unsigned int* __restrict__ sorted_queries,
    const unsigned int num_queries, 
    float* normals
);

} // namespace lbvh
} // namespace lvr2

#endif // LVR2_KNN_NORMALS_KERNEL_CUH