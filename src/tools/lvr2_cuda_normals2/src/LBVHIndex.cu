#include "LBVHIndex.cuh"
#include "lbvh_kernels.cuh"
#include "lbvh.cuh"

#include <stdio.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <thrust/sort.h>
#include <nvrtc.h>
#include <cuda.h>

#include "GPUErrorCheck.h"

using namespace lbvh;

LBVHIndex::LBVHIndex()
{
    this->m_num_objects = 0;
    this->m_num_nodes = 0;
    this->m_leaf_size = false;
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

    // thrust::sort_by_key(keys, keys + num_points, values);
    thrust::sort_by_key(h_morton_codes, h_morton_codes + num_points, 
                        aabbs);
    gpuErrchk(cudaPeekAtLastError());

    this->m_sorted_indices = h_morton_codes;

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

    gpuErrchk(cudaMemcpy(d_sorted_aabbs, aabbs, 
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

    cudaMemcpy(nodes, d_nodes, m_num_nodes * sizeof(BVHNode), 
                cudaMemcpyDeviceToHost);
    
    gpuErrchk(cudaPeekAtLastError());

    this->m_nodes = nodes;

    gpuErrchk(cudaMemcpy(root_node, d_root_node, 
            sizeof(unsigned int), cudaMemcpyDeviceToHost));
    
    // TODO: Root node might be wrong?
    this->m_root_node = root_node;

    printf("Root: %u \n", *this->m_root_node);

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

void LBVHIndex::process_queries(float* queries_raw, size_t num_queries, float* args, 
                    float* points_raw, size_t num_points,
                    const char* kernel)
{
    // const BVHNode *nodes,
    
    // const float3* __restrict__ points,
    float3* points = (float3*) malloc(sizeof(float3) * num_points);
    for(int i = 0; i < num_points; i++)
    {
        points[i].x = points_raw[3 * i + 0];
        points[i].y = points_raw[3 * i + 1];
        points[i].z = points_raw[3 * i + 2];
    }
    
    // TODO: Implement sorting of indices
    // const unsigned int* __restrict__ sorted_indices,
    unsigned int* sorted_indices = (unsigned int*) 
                malloc(sizeof(unsigned int) * num_queries);

    for(int i = 0; i < num_queries; i++)
    {
        sorted_indices[i] = i;
    }

    // const unsigned int root_index,
    // = this->m_root_node

    // const float max_radius,
    // TODO: Implement radius
    float max_radius = FLT_MAX;
    
    // const float3* __restrict__ query_points,
    float3* query_points = (float3*) malloc(sizeof(float3) * num_points);
    for(int i = 0; i < num_points; i++)
    {
        query_points[i].x = queries_raw[3 * i + 0];
        query_points[i].y = queries_raw[3 * i + 1];
        query_points[i].z = queries_raw[3 * i + 2];
    }

    // const unsigned int* __restrict__ sorted_queries,
    // = sorted_indices

    // const unsigned int N,
    // = num_queries
    
    // // custom parameters
    // unsigned int* indices_out,
    // float* distances_out,
    // unsigned int* n_neighbors_out

    // Get the ptx from the query_knn_kernels
    std::string ptx;

    // const char** log_string;

    getPtxFromCuString(ptx, "query_knn_kernels", kernel, NULL, NULL);

    std::cout << "PTX kernel: " << std::endl;
    std::cout << ptx << std::endl;

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

// #define NVRTC_CHECK_ERROR(func)                                                                                      \
//     do                                                                                                                 \
//     {                                                                                                                  \
//       nvrtcResult code = func;                                                                                         \
//       if (code != NVRTC_SUCCESS)                                                                                       \
//       {                                                                                                                \
//         std::cout << "Error in NVRTC" << std::string(nvrtcGetErrorString(code));                                       \
//         throw Exception("ERROR: " __FILE__ "(" LINE_STR "): " + std::string(nvrtcGetErrorString(code)));               \
//       }                                                                                                                \
//     } while (0)

#define NVRTC_SAFE_CALL(x)                                        \
  do {                                                            \
    nvrtcResult result = x;                                       \
    if (result != NVRTC_SUCCESS) {                                \
      std::cerr << "\nerror: " #x " failed with error "           \
                << nvrtcGetErrorString(result) << '\n';           \
      exit(1);                                                    \
    }                                                             \
  } while(0)

__host__
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
        "-DK=5"
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
    std::cout << compileRes << std::endl;
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