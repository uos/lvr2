#ifndef LBVHINDEX_CUH
#define LBVHINDEX_CUH

#include<string>

#include <cuda_runtime.h>

#include "aabb.cuh"
#include "lbvh.cuh"
#include "lbvh_kernels.cuh"

namespace lbvh
{
class LBVHIndex
{
public:
    unsigned int m_num_objects;
    unsigned int m_num_nodes;
    unsigned int m_leaf_size;
    bool m_sort_queries;
    bool m_compact;
    bool m_shrink_to_fit;

    float* m_points;
    unsigned long long int* m_sorted_indices;

    char* m_mode;
    float m_radius;
    AABB* m_extent;
    BVHNode* m_nodes;
    unsigned int* m_root_node;
    
    __host__
    LBVHIndex();

    __host__
    LBVHIndex(int leaf_size, bool sort_queries, bool compact, bool shrink_to_fit);

    __host__
    void build(float* points, size_t num_points);

    __host__
    void process_queries(float* queries_raw, size_t num_queries, 
                        float* args, float* points_raw, 
                        size_t num_points,
                        const char* kernel);
    
    __host__ 
    AABB* getExtent(AABB* extent, float* points, size_t num_points);

    __host__
    std::string getSampleDir();

    __host__
    void getPtxFromCuString( std::string& ptx, const char* sample_name, 
                                    const char* cu_source, const char* name, 
                                    const char** log_string );

};

}   // namespace lbvh

#endif // LBVHINDEX_CUH