#include "query_knn.cuh"
#include "eigenvectors.cuh"
#include "covariance.cuh"
#include "static_priorityqueue.cuh"

using namespace lbvh;

using KNNHandler = StaticPriorityQueue<float, K>;

__global__ void normals_knn_kernel(const BVHNode *nodes,
                                 const float3* __restrict__ points,
                                 const unsigned int* __restrict__ sorted_indices,
                                 unsigned int root_index,
                                 const float max_radius,
                                 const float3* __restrict__ query_points,
                                 const unsigned int* __restrict__ sorted_queries,
                                 unsigned int N,
                                 // custom arguments
                                 const unsigned short* k,
                                 float3* normals_out)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    unsigned int query_idx = sorted_queries[idx];
    CumulativeCovariance cumulant;
    {
        auto kq = k[query_idx];
        auto queue = query_knn(nodes, points, sorted_indices, root_index, &query_points[query_idx], max_radius);
        cumulant.update(queue, points, kq);
        cumulant.finalize();
    }
    normals_out[query_idx] = fast_min_eigenvector_3x3(cumulant.mat());
}
