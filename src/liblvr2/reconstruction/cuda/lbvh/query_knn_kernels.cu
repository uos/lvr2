#include "query_knn.cuh"

using namespace lvr2;
using namespace lbvh;

namespace lvr2
{

extern "C" __global__ void query_knn_kernel(
    const BVHNode *nodes,
    const float* __restrict__ points,          // Changed from float3* to float*
    const unsigned int* __restrict__ sorted_indices,
    const unsigned int root_index,
    const float max_radius,
    const float* __restrict__ query_points,    // Changed from float3* to float*
    const unsigned int* __restrict__ sorted_queries,
    const unsigned int N,
    // custom parameters
    unsigned int* indices_out,
    float* distances_out,
    unsigned int* n_neighbors_out
)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;
    StaticPriorityQueue<float, K> queue(max_radius);
    unsigned int query_idx = sorted_queries[idx];

    // added this
    for(int i = 0; i < K; i++)
    {
        indices_out[K * query_idx + i] = UINT_MAX;
        distances_out[K * query_idx + i] = FLT_MAX;
    }

    // added this
    float3 query_point =
    {
        query_points[3 * query_idx + 0],
        query_points[3 * query_idx + 1],
        query_points[3 * query_idx + 2]
    };
    // query_knn(nodes, points, sorted_indices, root_index, &query_points[query_idx], queue);
    query_knn(nodes, points, sorted_indices, root_index, &query_point, queue);
    __syncwarp(); // synchronize the warp before the write operation
    // write back the results at the correct position
    queue.write_results(&indices_out[query_idx * K], &distances_out[query_idx * K], &n_neighbors_out[query_idx]);
}

} // namespace lvr2
