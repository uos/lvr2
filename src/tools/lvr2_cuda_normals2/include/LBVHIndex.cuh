#ifndef LBVHINDEX_CUH
#define LBVHINDEX_CUH

#include <boost/filesystem.hpp>

#include "aabb.cuh"
#include "lbvh_kernels.cuh"

namespace lbvh
{

class LBVHIndex
{
public:
    int m_num_objects;
    int m_num_nodes;
    int m_leaf_size;
    bool m_sort_queries;
    bool m_compact;
    bool m_shrink_to_fit;

    float* m_points;
    int* m_sorted_indices;

    LBVHIndex(int leaf_size, bool sort_queries, bool compact, bool shrink_to_fit);

    void build(float* points, size_t num_points);

    __device__ __host__ AABB getExtent(float* points, size_t num_points);
};

}   // namespace lbvh

#endif // LBVHINDEX_CUH