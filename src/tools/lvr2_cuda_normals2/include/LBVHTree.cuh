#ifndef LBVHTREE_CUH
#define LBVHTREE_CUH

#include <string>

#include "LBVHIndex.cuh"

namespace lbvh
{
class LBVHTree
{
public:
    LBVHIndex tree;

    LBVHTree();

    LBVHTree(int leaf_size, bool sort_queries, bool compact, bool shrink_to_fit);

    void build(float* points, size_t num_points);

    // void prepare_knn();

    // void prepare_radius();

    void process_queries(float* queries_raw, size_t num_queries, 
                        float* args, float* points_raw, 
                        size_t num_points,
                        const char* cu_src,
                        const char* kernel_name,
                        int K);
    
    // AABB* getExtent(AABB* extent, float* points, size_t num_points);

    // std::string getSampleDir();

    // void getPtxFromCuString( std::string& ptx, const char* sample_name, 
    //                                 const char* cu_source, const char* name, 
    //                                 const char** log_string );

};

}   // namespace lbvh

#endif // LBVHTREE_CUH