#ifndef LBVHINDEX_CUH
#define LBVHINDEX_CUH

#include <string>

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
    bool m_shrink_to_fit;   // Probably not needed

    float* m_points;
    unsigned int* m_sorted_indices;

    char* m_mode;
    float m_radius;
    AABB* m_extent;
    BVHNode* m_nodes;
    unsigned int m_root_node;

    // TODO Do we need this here?
    float m_flip_x;
    float m_flip_y;
    float m_flip_z;
    
    LBVHIndex();

    LBVHIndex(int leaf_size, bool sort_queries, bool compact, bool shrink_to_fit, 
                        float flip_x=1000000.0f, float flip_y=1000000.0f, float flip_z=1000000.0f);

    void build(float* points, size_t num_points);

    void process_queries(float* queries_raw, size_t num_queries, 
                        float* args, float* points_raw, 
                        size_t num_points,
                        const char* cu_src,
                        const char* kernel_name,
                        int K,
                        unsigned int* n_neighbors_out, unsigned int* indices_out, float* distances_out);

    void calculate_normals(float* normals, size_t num_normals,
                        float* queries, size_t num_queries,
                        int K,
                        float* points, size_t num_points,
                        unsigned int* n_neighbors_out, unsigned int* indices_out);
    
    AABB* getExtent(AABB* extent, float* points, size_t num_points);

    std::string getSampleDir();

    void getPtxFromCuString( std::string& ptx, const char* sample_name, 
                                    const char* cu_source, const char* name, 
                                    const char** log_string );

};

}   // namespace lbvh

#endif // LBVHINDEX_CUH