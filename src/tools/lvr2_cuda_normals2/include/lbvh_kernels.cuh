#ifndef LBVH_KERNELS_CUH
#define LBVH_KERNELS_CUH

#define HASH_64 1 // use 64 bit morton codes

#include <cuda_runtime.h>

#include "aabb.cuh"
#include "lbvh.cuh"
#include "morton_code.cuh"

namespace lbvh {

// __global__
// void test_kernel();


__global__ 
void compute_morton_kernel(const AABB* __restrict__ aabbs, 
                        const AABB* __restrict__ extent, 
                        unsigned long long int* morton_codes, 
                        unsigned int N); 
   
__global__ 
void compute_morton_points_kernel(float* __restrict__ const points,     // Changed from float3* to float*
                                             AABB* __restrict__ const extent,
                                             unsigned long long int* morton_codes,
                                             unsigned int N);
    

__global__ 
void initialize_tree_kernel(BVHNode *nodes,
                            const AABB *sorted_aabbs,
                            unsigned int N);

__global__ 
void construct_tree_kernel(BVHNode *nodes,
                        unsigned int* root_index,
                        const unsigned long long int *sorted_morton_codes,
                        unsigned int N);


__global__ 
void optimize_tree_kernel(BVHNode *nodes,
                                     unsigned int* root_index,
                                     unsigned int* valid,
                                     unsigned int max_node_size,
                                     unsigned int N);

__global__ 
void compute_free_indices_kernel(const unsigned int* valid_sums,
                                     const unsigned int* isums,
                                     unsigned int* free_indices,
                                     unsigned int N);

__global__ 
void compact_tree_kernel(BVHNode *nodes,
                                     unsigned int* root_index,
                                     const unsigned int* valid_sums,
                                     const unsigned int* free_positions,
                                     unsigned int first_moved,
                                     unsigned int node_cnt_new,
                                     unsigned int N);
} // namespace lbvh

#endif // LBVH_KERNELS_CUH