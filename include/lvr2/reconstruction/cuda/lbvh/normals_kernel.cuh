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