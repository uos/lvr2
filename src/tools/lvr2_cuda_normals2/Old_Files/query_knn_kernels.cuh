#include "query_knn.cuh"

namespace lbvh
{
    
__global__ void query_knn_kernel(const BVHNode *nodes,
                                 const float3* __restrict__ points,
                                 const unsigned int* __restrict__ sorted_indices,
                                 const unsigned int root_index,
                                 const float max_radius,
                                 const float3* __restrict__ query_points,
                                 const unsigned int* __restrict__ sorted_queries,
                                 const unsigned int N,
                                 // custom parameters
                                 unsigned int* indices_out,
                                 float* distances_out,
                                 unsigned int* n_neighbors_out
                                 );
}