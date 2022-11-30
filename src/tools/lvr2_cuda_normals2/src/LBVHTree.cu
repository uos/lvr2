#include "LBVHTree.cuh"


using namespace lbvh;

LBVHTree::LBVHTree()
{
    this->tree = lbvh::LBVHIndex();
}

LBVHTree::LBVHTree(int leaf_size, bool sort_queries, bool compact, bool shrink_to_fit)
{
    this->tree = lbvh::LBVHIndex(leaf_size, sort_queries, compact, shrink_to_fit);
}

void LBVHTree::build(float* points, size_t num_points)
{
    this->tree.build(points, num_points);
}

// void prepare_knn();

// void prepare_radius();

void LBVHTree::process_queries(float* queries_raw, size_t num_queries, 
                        float* args, float* points_raw, 
                        size_t num_points,
                        const char* cu_src,
                        const char* kernel_name,
                        int K)
{
    this->tree.process_queries(queries_raw, num_queries, args, 
                        points_raw, num_points,
                        cu_src, kernel_name,
                        K);
}