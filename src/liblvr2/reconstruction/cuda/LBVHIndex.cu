#include "lvr2/reconstruction/cuda/LBVHIndex.hpp"
#include "lvr2/reconstruction/cuda/lbvh/lbvh_kernels.cuh"
#include "lvr2/reconstruction/cuda/lbvh/lbvh.cuh"
#include "lvr2/reconstruction/cuda/lbvh/normals_kernel.cuh"
#include "lvr2/reconstruction/cuda/lbvh/aabb.cuh"

#include <stdio.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <exception>
#include <string>
#include <thrust/sort.h>
#include <nvrtc.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "lvr2/reconstruction/cuda/lbvh/GPUErrorCheck.h"

using namespace lvr2;
using namespace lbvh;

namespace lvr2
{

LBVHIndex::LBVHIndex()
{
    this->m_num_objects = 0;
    this->m_num_nodes = 0;
    this->m_leaf_size = 1;
    this->m_sort_queries = true;
    this->m_compact = true;

    this->m_flip_x = 1000000.0;
    this->m_flip_y = 1000000.0;
    this->m_flip_z = 1000000.0;
    
}

LBVHIndex::LBVHIndex(
    int leaf_size, 
    bool sort_queries, 
    bool compact,
    float flip_x, 
    float flip_y, 
    float flip_z
)
{
    this->m_num_objects = 0;
    this->m_num_nodes = 0;
    this->m_leaf_size = leaf_size;
    this->m_sort_queries = sort_queries;
    this->m_compact = compact;

    this->m_flip_x = flip_x;
    this->m_flip_y = flip_y;
    this->m_flip_z = flip_z;
    
}

LBVHIndex::~LBVHIndex()
{
    // CPU
    free(this->m_root_node);

    // GPU
    cudaFree(this->m_d_points);
    cudaFree(this->m_d_sorted_indices);
    cudaFree(this->m_d_nodes);
    cudaFree(this->m_d_extent);

}

void LBVHIndex::build(float* points, size_t num_points)
{
    this->m_num_objects = num_points;
    this->m_num_nodes = 2 * m_num_objects - 1;

    // initialize AABBs
    AABB* aabbs = (struct AABB*) malloc(sizeof(struct AABB) * num_points);

    // Initial bounding boxes are the points
    for(int i = 0; i < m_num_objects; i ++)
    {
        aabbs[i].min.x = points[3 * i + 0];
        aabbs[i].max.x = points[3 * i + 0];
        aabbs[i].min.y = points[3 * i + 1];
        aabbs[i].max.y = points[3 * i + 1];
        aabbs[i].min.z = points[3 * i + 2];
        aabbs[i].max.z = points[3 * i + 2];
    }
    // Get the extent
    AABB* extent = (struct AABB*) malloc(sizeof(struct AABB));
    getExtent(extent, points, m_num_objects);
    
    gpuErrchk(cudaMalloc(&this->m_d_extent, sizeof(struct AABB)));
    gpuErrchk(cudaMemcpy(this->m_d_extent, extent, sizeof(struct AABB), cudaMemcpyHostToDevice));

    AABB* d_aabbs;
    gpuErrchk(cudaMalloc(&d_aabbs, sizeof(struct AABB) * num_points));
    gpuErrchk(cudaMemcpy(d_aabbs, aabbs, sizeof(struct AABB) * num_points, cudaMemcpyHostToDevice));

    int size_morton = num_points * sizeof(unsigned long long int);

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_points + threadsPerBlock - 1) 
                        / threadsPerBlock;

    // Get the morton codes of the points
    unsigned long long int* d_morton_codes;
    gpuErrchk(cudaMalloc(&d_morton_codes, size_morton));

    compute_morton_kernel<<<blocksPerGrid, threadsPerBlock>>>
        (d_aabbs, this->m_d_extent, d_morton_codes, num_points);
    
    gpuErrchk(cudaPeekAtLastError());
    
    gpuErrchk( cudaFree(d_aabbs) );
    
    unsigned long long int* morton_codes = (unsigned long long int*)
                    malloc(sizeof(unsigned long long int) * num_points);

    gpuErrchk( cudaMemcpy(morton_codes, d_morton_codes, size_morton, cudaMemcpyDeviceToHost) );
    
    gpuErrchk( cudaFree(d_morton_codes) );

    // Create array of indices with an index for each point
    unsigned int* indices = (unsigned int*)
        malloc(sizeof(unsigned int) * num_points);

    for(int i = 0; i < num_points; i++)
    {
        indices[i] = i;
    }

    // Sort the indices according to the corresponding morton codes
    thrust::sort_by_key(morton_codes, morton_codes + num_points, 
                        indices);
    
    // Sort the AABBs by the indices
    AABB* sorted_aabbs = (AABB*) malloc(sizeof(AABB) * num_points);
   
    for(int i = 0; i < num_points; i++)
    {
        sorted_aabbs[i] = aabbs[indices[i]];
    }

    gpuErrchk(cudaPeekAtLastError());

    // Create the nodes
    gpuErrchk(cudaMalloc(&this->m_d_nodes, sizeof(struct BVHNode) * m_num_nodes));

    AABB* d_sorted_aabbs;
    gpuErrchk(cudaMalloc(&d_sorted_aabbs, 
        sizeof(struct AABB) * num_points));

    gpuErrchk(cudaMemcpy(d_sorted_aabbs, sorted_aabbs, 
        sizeof(struct AABB) * num_points, cudaMemcpyHostToDevice));

    // Initialize the tree
    initialize_tree_kernel<<<blocksPerGrid, threadsPerBlock>>>
        (this->m_d_nodes, d_sorted_aabbs, num_points);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaFree(d_sorted_aabbs));

    // Construct the tree
    this->m_root_node = (unsigned int*)
        malloc(sizeof(unsigned int));
    this->m_root_node[0] = UINT_MAX;

    unsigned int* d_root_node;
    gpuErrchk(cudaMalloc(&d_root_node, sizeof(unsigned int)));

    gpuErrchk(cudaMemcpy(d_root_node, m_root_node, sizeof(unsigned int), 
                cudaMemcpyHostToDevice));

    unsigned long long int* d_sorted_morton_codes;
    gpuErrchk(cudaMalloc(&d_sorted_morton_codes, size_morton));

    gpuErrchk(cudaMemcpy(d_sorted_morton_codes, morton_codes, 
            size_morton, cudaMemcpyHostToDevice));

    construct_tree_kernel<<<blocksPerGrid, threadsPerBlock>>>
        (this->m_d_nodes, d_root_node, d_sorted_morton_codes, num_points);

    // Optimize the tree
    if(this->m_leaf_size > 1)
    {
        unsigned int* valid = (unsigned int*)
            malloc(sizeof(unsigned int) * this->m_num_nodes);

        for(int i = 0; i < this->m_num_nodes; i++)
        {
            valid[i] = 1;
        }

        unsigned int* d_valid;
        cudaMalloc(&d_valid, sizeof(unsigned int) * this->m_num_nodes);

        cudaMemcpy(d_valid, valid, 
            sizeof(unsigned int) * this->m_num_nodes,
            cudaMemcpyHostToDevice);

        optimize_tree_kernel<<<blocksPerGrid, threadsPerBlock>>>
            (this->m_d_nodes, d_root_node, d_valid, this->m_leaf_size, this->m_num_objects);

        cudaMemcpy(valid, d_valid, 
            sizeof(unsigned int) * this->m_num_nodes,
            cudaMemcpyDeviceToHost);

        // Compact tree to increase bandwidth
        if(this->m_compact)
        {
            // Get the cumulative sum of valid, but start with 0
            unsigned int* valid_sums = (unsigned int*)
                malloc(sizeof(unsigned int) * this->m_num_nodes + 1);

            valid_sums[0] = 0;
            for(int i = 1; i < this->m_num_nodes + 1; i++)
            {
                valid_sums[i] = valid_sums[i - 1] + valid[i - 1];
            }

            // Number of the actually used nodes after optimizing
            unsigned int new_node_count = valid_sums[this->m_num_nodes];

            // Calculate the isum parameter
            unsigned int* isum = (unsigned int*)
                malloc(sizeof(unsigned int) * this->m_num_nodes);

            for(int i = 0; i < this->m_num_nodes; i++)
            {
                isum[i] = i - valid_sums[i];
            }
            unsigned int free_indices_size = isum[new_node_count];

            unsigned int* free_indices;

            // Reuse valid space, since its not needed anymore
            free_indices = &valid[0];

            // Upload
            unsigned int* d_valid_sums;
            unsigned int* d_isum;
            unsigned int* d_free_indices;

            cudaMalloc(&d_valid_sums, sizeof(unsigned int) * this->m_num_nodes + 1);
            cudaMalloc(&d_isum, sizeof(unsigned int) * this->m_num_nodes);
            cudaMalloc(&d_free_indices, sizeof(unsigned int) * free_indices_size);

            cudaMemcpy(d_valid_sums, valid_sums,
                sizeof(unsigned int) * this->m_num_nodes + 1,
                cudaMemcpyHostToDevice);
            cudaMemcpy(d_isum, isum,
                sizeof(unsigned int) * this->m_num_nodes,
                cudaMemcpyHostToDevice);
            cudaMemcpy(d_free_indices, free_indices,
                sizeof(unsigned int) * free_indices_size,
                cudaMemcpyHostToDevice);
            
            int threadsPerBlock = 256;
            int blocksPerGrid = (new_node_count + threadsPerBlock - 1) 
                        / threadsPerBlock;

            compute_free_indices_kernel<<<blocksPerGrid, threadsPerBlock>>>
                (d_valid_sums, d_isum, d_free_indices, new_node_count);

            cudaMemcpy(valid_sums, d_valid_sums, 
                sizeof(unsigned int) * this->m_num_nodes + 1,
                cudaMemcpyDeviceToHost);

            // get the sum of the first object that has to be moved
            unsigned int first_moved = valid_sums[new_node_count];

            threadsPerBlock = 256;
            blocksPerGrid = (this->m_num_nodes + threadsPerBlock - 1) 
                        / threadsPerBlock;

            // self.nodes, root_node, valid_sums_aligned, free, first_moved, new_node_count, self.num_nodes
            compact_tree_kernel<<<blocksPerGrid, threadsPerBlock>>>
            (this->m_d_nodes, d_root_node, d_valid_sums, d_free_indices, first_moved, new_node_count, this->m_num_nodes);

            this->m_num_nodes = new_node_count;
            free(valid_sums);
            free(isum);

            gpuErrchk( cudaFree(d_valid_sums) );
            gpuErrchk( cudaFree(d_isum) );
            gpuErrchk( cudaFree(d_free_indices) );
        }
        gpuErrchk( cudaFree(d_valid) );

        free(valid);
    }
    
    gpuErrchk(cudaMemcpy(m_root_node, d_root_node, 
            sizeof(unsigned int), cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(d_root_node));
    gpuErrchk(cudaFree(d_sorted_morton_codes));

    
    // Upload points to GPU
    gpuErrchk( cudaMalloc(&this->m_d_points,
        sizeof(float) * 3 * num_points) );
    gpuErrchk( cudaMemcpy(this->m_d_points, points,
        sizeof(float) * 3 * num_points,
        cudaMemcpyHostToDevice) );
    // Upload sorted indices to GPU
    gpuErrchk( cudaMalloc(&this->m_d_sorted_indices,
        sizeof(unsigned int) * num_points) );
    gpuErrchk( cudaMemcpy(this->m_d_sorted_indices, indices,
        sizeof(unsigned int) * num_points,
        cudaMemcpyHostToDevice) );

    free(aabbs);
    free(morton_codes);
    free(indices);
    free(sorted_aabbs);

    return;
}

#define CUDA_SAFE_CALL(x) \
    do { \
        CUresult result = x; \
        if (result != CUDA_SUCCESS) { \
            const char *msg; \
            cuGetErrorName(result, &msg); \
            std::cerr << "\nerror: " #x " failed with error " \
            << msg << '\n'; \
            exit(1); \
        } \
    } while(0)

#define NVRTC_SAFE_CALL(x)                                        \
    do {                                                            \
        nvrtcResult result = x;                                       \
        if (result != NVRTC_SUCCESS) {                                \
            std::cerr << "\nerror: " #x " failed with error "           \
                        << nvrtcGetErrorString(result) << '\n';           \
            exit(1);                                                    \
        }                                                             \
    } while(0)

void LBVHIndex::kSearch(
    float* query_points, 
    size_t num_queries,
    int K, 
    unsigned int* n_neighbors_out, 
    unsigned int* indices_out, 
    float* distances_out
) const
{   
    float radius = FLT_MAX;
    
    this->process_queries(query_points, num_queries, K, radius,
        n_neighbors_out, indices_out, distances_out);
}

void LBVHIndex::kSearch_dev_ptr(
    float* d_query_points, 
    size_t num_queries,
    int K, 
    unsigned int* d_n_neighbors_out, 
    unsigned int* d_indices_out, 
    float* d_distances_out
) const
{
    float radius = FLT_MAX;
    
    this->process_queries_dev_ptr(d_query_points, num_queries, K, radius,
        d_n_neighbors_out, d_indices_out, d_distances_out);
}

void LBVHIndex::radiusSearch(
    float* query_points, 
    size_t num_queries,
    int K, 
    float r,
    unsigned int* n_neighbors_out, 
    unsigned int* indices_out, 
    float* distances_out
) const
{   
    this->process_queries(query_points, num_queries, K, r,
        n_neighbors_out, indices_out, distances_out);
}

void LBVHIndex::radiusSearch_dev_ptr(
    float* d_query_points, 
    size_t num_queries,
    int K, 
    float r,
    unsigned int* d_n_neighbors_out, 
    unsigned int* d_indices_out, 
    float* d_distances_out
) const
{
    this->process_queries_dev_ptr(d_query_points, num_queries, K, r,
        d_n_neighbors_out, d_indices_out, d_distances_out);
}

void LBVHIndex::process_queries(
    float* queries_raw, 
    size_t num_queries,
    int K, 
    float r,
    unsigned int* n_neighbors_out, 
    unsigned int* indices_out, 
    float* distances_out
) const
{
    unsigned int* d_n_neighbors_out; 
    unsigned int* d_indices_out; 
    float* d_distances_out;
    
    // Allocate output buffer
    gpuErrchk( cudaMalloc(&d_indices_out, sizeof(unsigned int) * num_queries * K) ); // Now here out of memory#########################
    gpuErrchk( cudaMalloc(&d_distances_out, sizeof(float) * num_queries * K) );
    gpuErrchk( cudaMalloc(&d_n_neighbors_out, sizeof(unsigned int) * num_queries) );

    // Upload
    float* d_query_points;
    gpuErrchk( cudaMalloc(&d_query_points, sizeof(float) * 3 * num_queries) );
    gpuErrchk( cudaMemcpy(d_query_points, queries_raw,
        sizeof(float) * 3 * num_queries,
        cudaMemcpyHostToDevice) );
    
    // Compute on GPU
    this->process_queries_dev_ptr(
        d_query_points, 
        num_queries,
        K, 
        r,
        d_n_neighbors_out, 
        d_indices_out, 
        d_distances_out
    );
    
    // Download
    gpuErrchk( cudaMemcpy(indices_out, d_indices_out,
            sizeof(unsigned int) * num_queries * K,
            cudaMemcpyDeviceToHost) );

    gpuErrchk( cudaMemcpy(distances_out, d_distances_out,
            sizeof(float) * num_queries * K,
            cudaMemcpyDeviceToHost) );

    gpuErrchk( cudaMemcpy(n_neighbors_out, d_n_neighbors_out,
            sizeof(unsigned int) * num_queries,
            cudaMemcpyDeviceToHost) );


    cudaFree(d_indices_out);
    cudaFree(d_distances_out);
    cudaFree(d_n_neighbors_out);
    cudaFree(d_query_points);
}

void LBVHIndex::process_queries_dev_ptr(
    float* d_query_points, 
    size_t num_queries,
    int K, 
    float r,
    unsigned int* d_n_neighbors_out, 
    unsigned int* d_indices_out, 
    float* d_distances_out
) const
{
    // Get the Query Kernel
    std::string kernel_file = "query_knn_kernels.cu";
    std::string kernel_name = "query_knn_kernel";
    std::string kernel_dir = std::string(LBVH_KERNEL_DIR);
    std::string kernel_path = kernel_dir + "/" + kernel_file;

    // Read the kernel file
    std::ifstream in(kernel_path);
    std::string cu_src((std::istreambuf_iterator<char>(in)),
        std::istreambuf_iterator<char>());

    // Get the ptx of the kernel
    std::string ptx_src;

    getPtxFromCuString(
        ptx_src, 
        kernel_name.c_str(), 
        cu_src.c_str(), 
        K
    );
    
    // Get cuda module and function
    CUmodule module;
    CUfunction kernel;
    CUDA_SAFE_CALL(cuInit(0));
    CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, ptx_src.c_str(), 0, 0, 0));
    CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, kernel_name.c_str()));

    // Prepare kernel launch
    unsigned int* sorted_queries = (unsigned int*) 
                malloc(sizeof(unsigned int) * num_queries);

    for(int i = 0; i < num_queries; i++)
    {
        sorted_queries[i] = i;
    }

    // Sort the queries according to their morton codes
    if(this->m_sort_queries)
    {
        unsigned long long int* morton_codes_query =
            (unsigned long long int*)
            malloc(sizeof(unsigned long long int) * num_queries);

        unsigned long long int* d_morton_codes_query;
        cudaMalloc(&d_morton_codes_query, 
            sizeof(unsigned long long int) * num_queries);

        int threadsPerBlock = 256;
        int blocksPerGrid = (num_queries + threadsPerBlock - 1) 
                        / threadsPerBlock;

        compute_morton_points_kernel<<<blocksPerGrid, threadsPerBlock>>>
            (d_query_points, this->m_d_extent, d_morton_codes_query, num_queries);

        cudaMemcpy(morton_codes_query, d_morton_codes_query,
            sizeof(unsigned long long int) * num_queries,
            cudaMemcpyDeviceToHost);
        
        thrust::sort_by_key(morton_codes_query, morton_codes_query + num_queries, 
                        sorted_queries);
    }

    // Upload
    unsigned int* d_sorted_queries;
    gpuErrchk( cudaMalloc(&d_sorted_queries, sizeof(unsigned int) * num_queries) );
    gpuErrchk( cudaMemcpy(d_sorted_queries, sorted_queries,
            sizeof(unsigned int) * num_queries,
            cudaMemcpyHostToDevice) );

    // Params need to be cast to const
    BVHNode* d_nodes = const_cast<BVHNode*>(this->m_d_nodes);
    float* d_points = const_cast<float*>(this->m_d_points);
    unsigned int* d_sorted_indices = const_cast<unsigned int*>(this->m_d_sorted_indices);
    unsigned int root_node = this->m_root_node[0];
    float radius = r;

    void *params[] = 
    {
        &d_nodes,
        &d_points,
        &d_sorted_indices,
        &root_node,
        &radius,
        &d_query_points,
        &d_sorted_queries,
        &num_queries,
        &d_indices_out,
        &d_distances_out,
        &d_n_neighbors_out
    };

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_queries + threadsPerBlock - 1) 
                        / threadsPerBlock;
   
    // Launch the kernel
    CUDA_SAFE_CALL( cuLaunchKernel(kernel, 
        blocksPerGrid, 1, 1,  // grid dim
        threadsPerBlock, 1, 1,    // block dim
        0, NULL,    // shared mem and stream
        params,       // arguments
        0
    ) );      

    cudaFree(d_query_points);
    cudaFree(d_sorted_queries);

    free(sorted_queries);
    
    return;
}

 void LBVHIndex::calculate_normals(
    float* normals, 
    size_t num_normals,
    float* queries, 
    size_t num_queries,
    int K,
    const unsigned int* n_neighbors_in, 
    const unsigned int* indices_in
)   const
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_normals + threadsPerBlock - 1) 
                        / threadsPerBlock;

    // Create device memory
    float* d_queries;
    gpuErrchk( cudaMalloc(&d_queries, 
        sizeof(float) * 3 * num_queries) );
    
    float* d_normals;
    gpuErrchk( cudaMalloc(&d_normals, 
        sizeof(float) * 3 * num_normals) );

    unsigned int* d_n_neighbors_in;
    gpuErrchk( cudaMalloc(&d_n_neighbors_in, 
        sizeof(unsigned int) * num_queries) );

    unsigned int* d_indices_in;
    gpuErrchk( cudaMalloc(&d_indices_in, 
        sizeof(unsigned int) * K * num_queries) );

    // Upload
    gpuErrchk( cudaMemcpy(d_queries, queries,
        sizeof(float) * 3 * num_queries,
        cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_n_neighbors_in, n_neighbors_in,
        sizeof(unsigned int) * num_queries, 
        cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_indices_in, indices_in,
        sizeof(unsigned int) * K * num_queries, 
        cudaMemcpyHostToDevice) );
    
    // Call the normals kernel
    calculate_normals_kernel<<<blocksPerGrid, threadsPerBlock>>>
        (this->m_d_points, d_queries, num_queries, K, d_n_neighbors_in, d_indices_in,
        d_normals, this->m_flip_x, this->m_flip_y, this->m_flip_z);
    
    // Download the normals
    gpuErrchk( cudaMemcpy(normals, d_normals,
        sizeof(float) * 3 * num_normals,
        cudaMemcpyDeviceToHost) );
    
    cudaFree(d_queries);
    cudaFree(d_normals);
    cudaFree(d_indices_in);
    cudaFree(d_n_neighbors_in);
}


void LBVHIndex::knn_normals(
    int K,
    float* normals, 
    size_t num_normals
)
{
    float* d_query_points = this->m_d_points;
    unsigned int* d_sorted_queries = this->m_d_sorted_indices;
    size_t num_queries = num_normals;
    
    // Get the KNN Normals Kernel
    std::string kernel_file = "knn_normals_kernel.cu";
    std::string kernel_name = "knn_normals_kernel";
    std::string kernel_dir = std::string(LBVH_KERNEL_DIR);
    std::string kernel_path = kernel_dir + "/" + kernel_file;
    
    // Read the kernel file
    std::ifstream in(kernel_path);
    std::string cu_src((std::istreambuf_iterator<char>(in)),
        std::istreambuf_iterator<char>());

    // Get the ptx of the kernel
    std::string ptx_src;

    getPtxFromCuString(
        ptx_src, 
        kernel_name.c_str(), 
        cu_src.c_str(), 
        K
    );

    // Init cuda
    cudaFree(0);
    
    // Get the cuda module and function
    CUmodule module;
    CUfunction kernel;

    CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, ptx_src.c_str(), 0, 0, 0));
    CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, kernel_name.c_str()));

    float* d_normals;
    gpuErrchk( cudaMalloc(&d_normals, 
        sizeof(float) * 3 * num_normals) );

    float radius = FLT_MAX;

    // Gather the arguments
    void *params[] = 
    {
        &this->m_d_nodes,
        &this->m_d_points,
        &this->m_d_sorted_indices,
        &this->m_root_node[0],
        &radius,
        &d_query_points,
        &d_sorted_queries,
        &num_queries,
        &d_normals
    };

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_queries + threadsPerBlock - 1) 
                        / threadsPerBlock;
   
    // Launch the kernel
    CUDA_SAFE_CALL( cuLaunchKernel(kernel, 
        blocksPerGrid, 1, 1,  // grid dim
        threadsPerBlock, 1, 1,    // block dim
        0, NULL,    // shared mem and stream
        params,       // arguments
        0
    ) );    

    gpuErrchk( cudaMemcpy(normals, d_normals,
        sizeof(float) * 3 * num_normals,
        cudaMemcpyDeviceToHost) );  
    
    cudaFree(d_sorted_queries);
    cudaFree(d_normals);
}

// Get the extent of the points 
// (minimum and maximum values in each dimension)
void LBVHIndex::getExtent(
    AABB* extent, 
    float* points, 
    size_t num_points
) const
{
    float min_x = INT_MAX;
    float min_y = INT_MAX;
    float min_z = INT_MAX;

    float max_x = INT_MIN;
    float max_y = INT_MIN;
    float max_z = INT_MIN;

    for(int i = 0; i < num_points; i++)
    {
        if(points[3 * i + 0] < min_x)
        {
            min_x = points[3 * i + 0];
        }

        if(points[3 * i + 1] < min_y)
        {
            min_y = points[3 * i + 1];
        }

        if(points[3 * i + 2] < min_z)
        {
            min_z = points[3 * i + 2];
        }

        if(points[3 * i + 0] > max_x)
        {
            max_x = points[3 * i + 0];
        }

        if(points[3 * i + 1] > max_y)
        {
            max_y = points[3 * i + 1];
        }

        if(points[3 * i + 2] > max_z)
        {
            max_z = points[3 * i + 2];
        }
    }
    
    extent->min.x = min_x;
    extent->min.y = min_y;
    extent->min.z = min_z;
    
    extent->max.x = max_x;
    extent->max.y = max_y;
    extent->max.z = max_z;
    
    return;
}

void LBVHIndex::getPtxFromCuString( 
    std::string& ptx, 
    const char* sample_name, 
    const char* cu_source, 
    int K
) const
{
    // Create program
    nvrtcProgram prog;
    NVRTC_SAFE_CALL( nvrtcCreateProgram( &prog, cu_source, sample_name, 0, NULL, NULL ) );

    std::string K_str = "-DK=" + std::to_string(K); 

    // Gather NVRTC options
    std::string kernel_includes = std::string("-I") + 
        std::string(LBVH_KERNEL_INCLUDES);

    std::string cuda_include = std::string("-I") + 
        std::string(CUDA_INCLUDE_DIRS);
    
    std::vector<const char*> options = {
        kernel_includes.c_str(),
        cuda_include.c_str(),
        "-std=c++17",
        K_str.c_str()
    };

    const std::string base_dir = std::string(LBVH_KERNEL_DIR);

    // JIT compile CU to PTX
    const nvrtcResult compileRes = nvrtcCompileProgram( prog, (int)options.size(), options.data() );
    
    // Retrieve log output
    size_t log_size = 0;
    NVRTC_SAFE_CALL( nvrtcGetProgramLogSize( prog, &log_size ) );

    char* log = new char[log_size];
    if( log_size > 1 )
    {
        NVRTC_SAFE_CALL( nvrtcGetProgramLog( prog, log ) );
        std::cout << log << std::endl;
    }
    
    if( compileRes != NVRTC_SUCCESS )
        throw std::runtime_error( "NVRTC Compilation failed.\n");

    // Retrieve PTX code
    size_t ptx_size = 0;
    NVRTC_SAFE_CALL( nvrtcGetPTXSize( prog, &ptx_size ) );
    ptx.resize( ptx_size );
    NVRTC_SAFE_CALL( nvrtcGetPTX( prog, &ptx[0] ) );

    // Cleanup
    NVRTC_SAFE_CALL( nvrtcDestroyProgram( &prog ) );
}

} // namespace lvr2
