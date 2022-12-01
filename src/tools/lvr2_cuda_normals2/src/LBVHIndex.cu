#include "LBVHIndex.cuh"
#include "lbvh_kernels.cuh"
#include "lbvh.cuh"
#include "normals_kernel.cuh"

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

#include "GPUErrorCheck.h"

using namespace lbvh;

// Only for testing
float quadratic_distance(float p1, float p2, float p3, float q1, float q2, float q3)
{
    return (p1 - q1) * (p1 - q1) + (p2 - q2) * (p2 - q2) + (p3 - q3) * (p3 - q3);
}

// Only for testing
void findKNN(int k, float* points, size_t num_points, float* queries, size_t num_queries)
{
    std::cout << "Brute forcing KNN..." << std::endl;
    float neighs[num_queries][k];

    float distances[num_queries][num_points];

    unsigned int indices[num_queries][num_points];

    for(int j = 0; j < num_queries; j++)
    {
        for(int i = 0; i < num_points; i++)
        {
            indices[j][i] = i;
        }

    }

    for(int i = 0; i < num_queries; i++)
    {
        for(int j = 0; j < num_points; j++)
        {
            distances[i][j] = quadratic_distance(
                                    points[3 * j + 0],
                                    points[3 * j + 1],
                                    points[3 * j + 2],
                                    queries[3 * i + 0],
                                    queries[3 * i + 1],
                                    queries[3 * i + 2]);
        }
    }
    for(int i = 0; i < num_queries; i++)
    {
        thrust::sort_by_key(distances[i], distances[i] + num_points, indices[i]);

    }

    for(int i = 0; i < num_queries; i++)
    {
        std::cout << "Query " << i << ": " << std::endl;
        std::cout << "Neighbors: " << std::endl;
        for(int j = 0; j < k; j++){
            std::cout << indices[i][j] << std::endl;
        }
        std::cout << "Distances: " << std::endl;
        for(int j = 0; j < k; j++)
        {
            std::cout << distances[i][j] << std::endl;
        }
    }
}


LBVHIndex::LBVHIndex()
{
    this->m_num_objects = 0;
    this->m_num_nodes = 0;
    this->m_leaf_size = 1;
    this->m_sort_queries = false;
    this->m_compact = false;
    this->m_shrink_to_fit = false;
    
}

LBVHIndex::LBVHIndex(int leaf_size, bool sort_queries, 
                    bool compact, bool shrink_to_fit)
{
    this->m_num_objects = 0;
    this->m_num_nodes = 0;
    this->m_leaf_size = leaf_size;
    this->m_sort_queries = sort_queries;
    this->m_compact = compact;
    this->m_shrink_to_fit = shrink_to_fit;

}

void LBVHIndex::build(float* points, size_t num_points)
{
    this->m_points = points;

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

    this->m_extent = extent;

    AABB* d_extent;
    gpuErrchk(cudaMalloc(&d_extent, sizeof(struct AABB)));
    //cudaMalloc(&d_extent, sizeof(struct AABB));
    
    gpuErrchk(cudaMemcpy(d_extent, extent, sizeof(struct AABB), cudaMemcpyHostToDevice));
    //cudaMemcpy(d_extent, extent, sizeof(struct AABB), cudaMemcpyHostToDevice);
    
    AABB* d_aabbs;
    gpuErrchk(cudaMalloc(&d_aabbs, sizeof(struct AABB) * num_points));
    //cudaMalloc(&d_aabbs, sizeof(struct AABB) * num_points);
    
    gpuErrchk(cudaMemcpy(d_aabbs, aabbs, sizeof(struct AABB) * num_points, cudaMemcpyHostToDevice));
    //cudaMemcpy(d_aabbs, aabbs, sizeof(struct AABB) * num_points, cudaMemcpyHostToDevice);

    int size_morton = num_points * sizeof(unsigned long long int);

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_points + threadsPerBlock - 1) 
                        / threadsPerBlock;

    // Get the morton codes of the points
    unsigned long long int* d_morton_codes;
    gpuErrchk(cudaMalloc(&d_morton_codes, size_morton));
    // cudaMalloc(&d_morton_codes, size_morton);

    compute_morton_kernel<<<blocksPerGrid, threadsPerBlock>>>
            (d_aabbs, d_extent, d_morton_codes, num_points);
    
    gpuErrchk(cudaPeekAtLastError());
    
    cudaFree(d_aabbs);
    cudaFree(d_extent);

    gpuErrchk(cudaDeviceSynchronize());
    // cudaDeviceSynchronize();

    unsigned long long int* h_morton_codes = (unsigned long long int*)
                    malloc(sizeof(unsigned long long int) * num_points);

    
    cudaMemcpy(h_morton_codes, d_morton_codes, size_morton, cudaMemcpyDeviceToHost);
    
    cudaFree(d_morton_codes);


    // Create array of indices with an index for each point
    unsigned int* indices = (unsigned int*)
        malloc(sizeof(unsigned int) * num_points);

    for(int i = 0; i < num_points; i++)
    {
        indices[i] = i;
    }

    // Sort the indices according to the corresponding morton code
    // thrust::sort_by_key(keys, keys + num_points, values);
    thrust::sort_by_key(h_morton_codes, h_morton_codes + num_points, 
                        indices);
    
    // Sort the AABBs by the indices
    AABB* sorted_aabbs = (AABB*) malloc(sizeof(AABB) * num_points);
    // TODO: Sort AABBs with thrust?
    // thrust::sort_by_key(h_morton_codes, h_morton_codes + num_points,
    //                  sorted_aabbs)

    for(int i = 0; i < num_points; i++)
    {
        sorted_aabbs[i] = aabbs[ indices[i] ];
    }

    gpuErrchk(cudaPeekAtLastError());

    this->m_sorted_indices = indices;


    // for(int i = 0; i < num_points - 1; i++)
    // {
    //     if(h_morton_codes[i] > h_morton_codes[i + 1])
    //     {
    //         printf("Error in sorting \n");
    //         break;
    //     }
    // }
    
    // Create the nodes
    BVHNode* nodes =  (struct BVHNode*) 
                    malloc(sizeof(struct BVHNode) * m_num_nodes); 

    BVHNode* d_nodes;
    gpuErrchk(cudaMalloc(&d_nodes, sizeof(struct BVHNode) * m_num_nodes));

    AABB* d_sorted_aabbs;
    gpuErrchk(cudaMalloc(&d_sorted_aabbs, 
            sizeof(struct AABB) * num_points));

    gpuErrchk(cudaMemcpy(d_sorted_aabbs, sorted_aabbs, 
            sizeof(struct AABB) * num_points, cudaMemcpyHostToDevice));

    // Initialize the tree
    initialize_tree_kernel<<<blocksPerGrid, threadsPerBlock>>>
        (d_nodes, d_sorted_aabbs, num_points);

    gpuErrchk(cudaPeekAtLastError());
    // gpuErrchk(cudaFree(0));

    // Construct the tree
    unsigned int* root_node = (unsigned int*)
        malloc(sizeof(unsigned int));
    *root_node = UINT_MAX;

    unsigned int* d_root_node;
    gpuErrchk(cudaMalloc(&d_root_node, sizeof(unsigned int)));

    gpuErrchk(cudaMemcpy(d_root_node, root_node, sizeof(unsigned int), 
                cudaMemcpyHostToDevice));

    unsigned long long int* d_sorted_morton_codes;
    gpuErrchk(cudaMalloc(&d_sorted_morton_codes, size_morton));

    gpuErrchk(cudaMemcpy(d_sorted_morton_codes, h_morton_codes, 
            size_morton, cudaMemcpyHostToDevice));

    construct_tree_kernel<<<blocksPerGrid, threadsPerBlock>>>
        (d_nodes, d_root_node, d_sorted_morton_codes, num_points);

    std::cout << "Old nodes number: " << this->m_num_nodes << std::endl;
    // Optimize the tree
    if(this->m_leaf_size > 1)
    {
        std::cout << "Optimizing Tree" << std::endl;
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
            (d_nodes, d_root_node, d_valid, this->m_leaf_size, this->m_num_objects);

        cudaMemcpy(valid, d_valid, 
            sizeof(unsigned int) * this->m_num_nodes,
            cudaMemcpyDeviceToHost);

        // Compact tree to increase bandwidth
        if(this->m_compact)
        {
            std::cout << "Compacting Tree" << std::endl;
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
            
            // // Leave out the last element again
            // unsigned int* valid_sums_aligned = (unsigned int*)
            //     malloc(sizeof(unsigned int) * this->m_num_nodes);

            // // TODO: Does this work this way?
            // valid_sums_aligned = &valid_sums[0];

            // Calculate the isum parameter
            unsigned int* isum = (unsigned int*)
                malloc(sizeof(unsigned int) * this->m_num_nodes);

            for(int i = 0; i < this->m_num_nodes; i++)
            {
                // isum[i] = i - valid_sums_aligned[i];
                isum[i] = i - valid_sums[i];
            }
            // Reuse valid space, since it not needed anymore
            unsigned int free_indices_size = isum[new_node_count];

            unsigned int* free = (unsigned int*)
                malloc(sizeof(unsigned int) * free_indices_size);

            free = &valid[0];

            // valid_sums, isum, free, new_node_count
            unsigned int* d_valid_sums;
            unsigned int* d_isum;
            unsigned int* d_free;

            cudaMalloc(&d_valid_sums, sizeof(unsigned int) * this->m_num_nodes + 1);
            cudaMalloc(&d_isum, sizeof(unsigned int) * this->m_num_nodes);
            cudaMalloc(&d_free, sizeof(unsigned int) * free_indices_size);

            cudaMemcpy(d_valid_sums, valid_sums,
                sizeof(unsigned int) * this->m_num_nodes + 1,
                cudaMemcpyHostToDevice);
            cudaMemcpy(d_isum, isum,
                sizeof(unsigned int) * this->m_num_nodes,
                cudaMemcpyHostToDevice);
            cudaMemcpy(d_free, free,
                sizeof(unsigned int) * free_indices_size,
                cudaMemcpyHostToDevice);
            
            int threadsPerBlock = 256;
            int blocksPerGrid = (new_node_count + threadsPerBlock - 1) 
                        / threadsPerBlock;

            compute_free_indices_kernel<<<blocksPerGrid, threadsPerBlock>>>
                (d_valid_sums, d_isum, d_free, new_node_count);

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
            (d_nodes, d_root_node, d_valid_sums, d_free, first_moved, new_node_count, this->m_num_nodes);

            this->m_num_nodes = new_node_count;
        }

    }
    std::cout << "Done!" << std::endl;
    std::cout << "New nodes number: " << this->m_num_nodes << std::endl;

    cudaMemcpy(nodes, d_nodes, m_num_nodes * sizeof(BVHNode), 
                cudaMemcpyDeviceToHost);

    this->m_nodes = nodes;

    gpuErrchk(cudaMemcpy(root_node, d_root_node, 
            sizeof(unsigned int), cudaMemcpyDeviceToHost));
    
    this->m_root_node = root_node[0];
    printf("Root: %u \n", this->m_root_node);

    gpuErrchk(cudaFree(d_root_node));
    gpuErrchk(cudaFree(d_sorted_aabbs));
    gpuErrchk(cudaFree(d_sorted_morton_codes));
    gpuErrchk(cudaFree(d_nodes));
    
    // free(root_node);
    // free(nodes);
    // free(aabbs);
    // free(extent);
    // free(h_morton_codes);

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

void LBVHIndex::process_queries(float* queries_raw, size_t num_queries, float* args, 
                    float* points_raw, size_t num_points,
                    const char* cu_src, const char* kernel_name,
                    int K,
                    unsigned int* n_neighbors_out, unsigned int* indices_out, float* distances_out)
{
    // Get the ptx of the kernel
    std::string ptx_src;

    // const char** log_string;

    getPtxFromCuString(ptx_src, kernel_name, cu_src, NULL, NULL);

    // std::cout << "PTX kernel: " << std::endl;
    // std::cout << ptx_src << std::endl;

    // Init cuda
    cudaFree(0);
    // CUdevice cuDevice;
    // CUcontext context;
    CUmodule module;
    CUfunction kernel;

    // CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));

    // CUDA_SAFE_CALL(cuCtxCreate(&context, 0, cuDevice));
    CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, ptx_src.c_str(), 0, 0, 0));
    CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, kernel_name));

    // Prepare kernel launch

    // const BVHNode *nodes,
    // this->m_nodes
    BVHNode* d_nodes;
    gpuErrchk( cudaMalloc(&d_nodes, sizeof(BVHNode) * this->m_num_nodes) );

    gpuErrchk( cudaMemcpy(d_nodes, this->m_nodes, 
            sizeof(BVHNode) * this->m_num_nodes,
            cudaMemcpyHostToDevice) );

    // const float3* __restrict__ points,
    float3* points3 = (float3*) malloc(sizeof(float3) * num_points);
    for(int i = 0; i < num_points; i++)
    {
        points3[i].x = points_raw[3 * i + 0];
        points3[i].y = points_raw[3 * i + 1];
        points3[i].z = points_raw[3 * i + 2];
    }

    float3* d_points3;
    gpuErrchk( cudaMalloc(&d_points3, sizeof(float3) * num_points) );

    gpuErrchk( cudaMemcpy(d_points3, points3, 
            sizeof(float3) * num_points, 
            cudaMemcpyHostToDevice) );
    
    // const unsigned int* __restrict__ sorted_indices,
    // this->m_sorted_indices
    unsigned long long int* d_sorted_indices;
    gpuErrchk( cudaMalloc(&d_sorted_indices, sizeof(unsigned int) * num_points) );

    gpuErrchk( cudaMemcpy(d_sorted_indices, this->m_sorted_indices,
            sizeof(unsigned int) * num_points,
            cudaMemcpyHostToDevice) );

    // const unsigned int root_index,
    // = this->m_root_node
    // unsigned int* d_root_node;
    // gpuErrchk( cudaMalloc(&d_root_node, sizeof(unsigned int)) );

    // gpuErrchk( cudaMemcpy(d_root_node, this->m_root_node,
    //         sizeof(unsigned int),
    //         cudaMemcpyHostToDevice) );

    // const float max_radius,
    // TODO: Implement radius
    this->m_radius = FLT_MAX;

    // float* max_radius = (float*) malloc(sizeof(float));
    // *max_radius = FLT_MAX;

    // float* d_max_radius;
    // gpuErrchk( cudaMalloc(&d_max_radius, sizeof(float)) );

    // gpuErrchk( cudaMemcpy(d_max_radius, max_radius,
    //         sizeof(float),
    //         cudaMemcpyHostToDevice) );
    
    // const float3* __restrict__ query_points,
    float3* query_points = (float3*) malloc(sizeof(float3) * num_queries);
    for(int i = 0; i < num_queries; i++)
    {
        query_points[i].x = queries_raw[3 * i + 0];
        query_points[i].y = queries_raw[3 * i + 1];
        query_points[i].z = queries_raw[3 * i + 2];
    }

    float3* d_query_points;
    gpuErrchk( cudaMalloc(&d_query_points, sizeof(float3) * num_queries) );

    gpuErrchk( cudaMemcpy(d_query_points, query_points,
            sizeof(float3) * num_queries,
            cudaMemcpyHostToDevice) );

    // const unsigned int* __restrict__ sorted_queries,
    unsigned int* sorted_queries = (unsigned int*) 
                malloc(sizeof(unsigned int) * num_queries);

    for(int i = 0; i < num_queries; i++)
    {
        sorted_queries[i] = i;
    }

    // only for large queries: sort them in morton order to prevent too much warp divergence on tree traversal
    if(this->m_sort_queries)
    {
        // queries, self.extent, morton_codes, queries.shape[0])
        AABB* d_extent;
        gpuErrchk(cudaMalloc(&d_extent, sizeof(struct AABB)));
        //cudaMalloc(&d_extent, sizeof(struct AABB));
    
        gpuErrchk(cudaMemcpy(d_extent, this->m_extent, sizeof(struct AABB), cudaMemcpyHostToDevice));
        //cudaMemcpy(d_extent, extent, sizeof(struct AABB), cudaMemcpyHostToDevice);

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
            (d_query_points, d_extent, d_morton_codes_query, num_queries);

        cudaMemcpy(morton_codes_query, d_morton_codes_query,
            sizeof(unsigned long long int) * num_queries,
            cudaMemcpyDeviceToHost);
        
        thrust::sort_by_key(morton_codes_query, morton_codes_query + num_queries, 
                        sorted_queries);
    }

    unsigned int* d_sorted_queries;
    gpuErrchk( cudaMalloc(&d_sorted_queries, sizeof(unsigned int) * num_queries) );

    gpuErrchk( cudaMemcpy(d_sorted_queries, sorted_queries,
            sizeof(unsigned int) * num_queries,
            cudaMemcpyHostToDevice) );

    // const unsigned int N,
    // = num_queries
    // size_t* N = (size_t*) malloc(sizeof(size_t));
    // *N = num_queries;

    // size_t* d_N;
    // gpuErrchk( cudaMalloc(&d_N, sizeof(size_t)) );

    // gpuErrchk( cudaMemcpy(d_N, N, 
    //         sizeof(size_t), 
    //         cudaMemcpyHostToDevice) );
    
    // // custom parameters
    // unsigned int* indices_out,
    for(int i = 0; i < num_queries * K; i++)
    {
        indices_out[i] = UINT32_MAX;
    }

    unsigned int* d_indices_out;
    gpuErrchk( cudaMalloc(&d_indices_out, sizeof(unsigned int) * num_queries * K) );

    gpuErrchk( cudaMemcpy(d_indices_out, indices_out,
            sizeof(unsigned int) * num_queries * K,
            cudaMemcpyHostToDevice) );


    // float* distances_out,
    for(int i = 0; i < num_queries * K; i++)
    {
        distances_out[i] = FLT_MAX;
    }

    float* d_distances_out;
    gpuErrchk( cudaMalloc(&d_distances_out, sizeof(float) * num_queries * K) );

    gpuErrchk( cudaMemcpy(d_distances_out, distances_out,
            sizeof(float) * num_queries * K,
            cudaMemcpyHostToDevice) );


    // unsigned int* n_neighbors_out
    unsigned int* d_n_neighbors_out;
    gpuErrchk( cudaMalloc(&d_n_neighbors_out, sizeof(unsigned int) * num_queries) );

     cudaMemcpy(this->m_sorted_indices, d_sorted_indices,
            sizeof(unsigned int) * num_points,
            cudaMemcpyDeviceToHost);

    // Gather the arguments
    void *params[] = 
    {
        &d_nodes,
        &d_points3,
        &d_sorted_indices,
        &this->m_root_node,
        &this->m_radius,
        &d_query_points,
        &d_sorted_queries,
        &num_queries,
        &d_indices_out,
        &d_distances_out,
        &d_n_neighbors_out
    };

    std::cout << "Launching Kernel" << std::endl;

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

    //CUDA_SAFE_CALL(cuCtxSynchronize());

    std::cout << "Kernel done, transferring memory..." << std::endl;
    
    gpuErrchk( cudaMemcpy(indices_out, d_indices_out,
            sizeof(unsigned int) * num_queries * K,
            cudaMemcpyDeviceToHost) );

    gpuErrchk( cudaMemcpy(distances_out, d_distances_out,
            sizeof(float) * num_queries * K,
            cudaMemcpyDeviceToHost) );

    gpuErrchk( cudaMemcpy(n_neighbors_out, d_n_neighbors_out,
            sizeof(unsigned int) * num_queries,
            cudaMemcpyDeviceToHost) );

    // findKNN(K, points_raw, num_points, queries_raw, num_queries);

}

 void LBVHIndex::calculate_normals(float* normals, size_t num_normals,
                        float* queries, size_t num_queries,
                        int K,
                        float* points, size_t num_points,
                        unsigned int* n_neighbors_out, unsigned int* indices_out)
{
    // Save the sum of all previous neighbors, important for indexing for radius search
    // (not necessarily K neighbors per query)
    unsigned int* neigh_sum = (unsigned int*) malloc(sizeof(unsigned int) * num_queries);

    neigh_sum[0] = 0;
    for(int i = 1; i < num_queries; i++)
    {
        neigh_sum[i] = neigh_sum[i - 1] + n_neighbors_out[i - 1];
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_normals + threadsPerBlock - 1) 
                        / threadsPerBlock;

    // Create device memory
    float* d_points;
    gpuErrchk( cudaMalloc(&d_points,
        sizeof(float) * 3 * num_points));

    float* d_queries;
    gpuErrchk( cudaMalloc(&d_queries, 
        sizeof(float) * 3 * num_queries) );

    unsigned int* d_n_neighbors_out;
    gpuErrchk( cudaMalloc(&d_n_neighbors_out, 
        sizeof(unsigned int) * num_queries) );

    unsigned int* d_indices_out;
    gpuErrchk( cudaMalloc(&d_indices_out, 
        sizeof(unsigned int) * K * num_queries) );

    unsigned int* d_neigh_sum;
    gpuErrchk( cudaMalloc(&d_neigh_sum, 
        sizeof(unsigned int) * num_queries) );

    float* d_normals;
    gpuErrchk( cudaMalloc(&d_normals, 
        sizeof(float) * 3 * num_normals) );

    // Copy to device
    gpuErrchk( cudaMemcpy(d_points, points,
        sizeof(float) * 3 * num_points,
        cudaMemcpyHostToDevice));

    gpuErrchk( cudaMemcpy(d_queries, queries,
        sizeof(float) * 3 * num_queries,
        cudaMemcpyHostToDevice) );

    gpuErrchk( cudaMemcpy(d_n_neighbors_out, n_neighbors_out,
        sizeof(unsigned int) * num_queries, 
        cudaMemcpyHostToDevice) );

    gpuErrchk( cudaMemcpy(d_indices_out, indices_out,
        sizeof(unsigned int) * K * num_queries, 
        cudaMemcpyHostToDevice) );
    
    gpuErrchk( cudaMemcpy(d_neigh_sum, neigh_sum, 
        sizeof(unsigned int) * num_queries, 
        cudaMemcpyHostToDevice) );

    gpuErrchk( cudaMemcpy(d_normals, normals,
        sizeof(float) * 3 * num_normals,
        cudaMemcpyHostToDevice) );
    
    calculate_normals_kernel<<<blocksPerGrid, threadsPerBlock>>>
        (d_points, d_queries, num_queries, d_n_neighbors_out, d_indices_out, d_neigh_sum, 
        d_normals);

    cudaDeviceSynchronize();

    gpuErrchk( cudaMemcpy(normals, d_normals,
        sizeof(float) * 3 * num_normals,
        cudaMemcpyDeviceToHost) );

}

// Get the extent of the points 
// (minimum and maximum values in each dimension)
AABB* LBVHIndex::getExtent(AABB* extent, float* points, size_t num_points)
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
    
    return extent;
}

std::string LBVHIndex::getSampleDir()
{
    // TODO: Don't use hard coded path
    return std::string("/home/till/Develop/src/tools/lvr2_cuda_normals2/src");
}
                         // Rückgabe String // Bsp: square_kernel.cu  // Inhalt d Datei     //Name Programm = NULL
void LBVHIndex::getPtxFromCuString( std::string& ptx, const char* sample_name, const char* cu_source, const char* name, const char** log_string )
{
    // Create program
    nvrtcProgram prog;
    NVRTC_SAFE_CALL( nvrtcCreateProgram( &prog, cu_source, sample_name, 0, NULL, NULL ) );

    // Gather NVRTC options
    std::string cuda_include = std::string("-I") + std::string(CUDA_INCLUDE_DIRS);
    std::vector<const char*> options = {
        "-I/home/till/Develop/src/tools/lvr2_cuda_normals2/include",
        cuda_include.c_str(),
        "-std=c++17",
        "-DK=50"
    };
    //      "-I/usr/local/cuda/include",
    //      "-I/usr/local/include",
    //      "-I/usr/include/x86_64-linux-gnu",
    //      "-I/usr/include",
    //      "-I/home/amock/workspaces/lvr/Develop/src/tools/lvr2_cuda_normals2/include"

    const std::string base_dir = getSampleDir();

    // JIT compile CU to PTX
    const nvrtcResult compileRes = nvrtcCompileProgram( prog, (int)options.size(), options.data() );
    // const char *options2[] = {"-I/home/till/Develop/src/tools/lvr2_cuda_normals2/src"};
    // const nvrtcResult compileRes = nvrtcCompileProgram( prog, 1, options );
    // std::cout << compileRes << std::endl;
    // Retrieve log output
    size_t log_size = 0;
    NVRTC_SAFE_CALL( nvrtcGetProgramLogSize( prog, &log_size ) );

    char* log = new char[log_size];
    if( log_size > 1 )
    {
        NVRTC_SAFE_CALL( nvrtcGetProgramLog( prog, log ) );
        // if( log_string )
        //     *log_string = log.c_str();
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