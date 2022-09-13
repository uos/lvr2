// #include <boost/filesystem.hpp>
// #include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>

// #include "lvr2/io/ModelFactory.hpp"
// #include "lvr2/util/Timestamp.hpp"
// #include "lvr2/util/IOUtils.hpp"
// #include "Options.hpp"
#include "CudaNormals.cuh"

struct MyNormal
{
    float x;
    float y;
    float z;
};



/*__global__
void initNormals_kernel(float* normals, size_t num_points)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_points)
    {
        normals[i * 3 + 0] = i + 0.0;
        normals[i * 3 + 1] = i + 1.0;
        normals[i * 3 + 2] = i + 2.0;
    }    
}

void initNormals(float* h_normals, size_t num_points)
{
    int num_bytes = num_points * 3 * sizeof(float);

    float* d_normals;
    cudaMalloc(&d_normals, num_bytes);

    // Get block and grid size
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_points + threadsPerBlock - 1) / threadsPerBlock;

    printf("%d %d \n", threadsPerBlock, blocksPerGrid);

    // Call the kernel
    initNormals_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_normals, num_points);

    cudaDeviceSynchronize();

    // Copy the normals back to host
    cudaMemcpy(h_normals, d_normals, num_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_normals);
} */

__global__
void initNormals2_kernel(MyNormal* normals, size_t num_points)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_points)
    {
        normals[i].x = i + 0.0;
        normals[i].y = i + 1.0;
        normals[i].z = i + 2.0;
    }    
}


void initNormals2(float* h_normals, size_t num_points)
{
    int num_bytes = num_points * sizeof(MyNormal);

    MyNormal* d_normals;
    cudaMalloc(&d_normals, num_bytes);

    // Initialize the normals
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_points + threadsPerBlock - 1) / threadsPerBlock;

    printf("%d %d \n", threadsPerBlock, blocksPerGrid);

    initNormals2_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_normals, num_points);

    cudaDeviceSynchronize();

    // Copy the normals back to host
    cudaMemcpy(h_normals, d_normals, num_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_normals);
}

//void getMortonCodes(float* h_normals, size_t num_points, )
