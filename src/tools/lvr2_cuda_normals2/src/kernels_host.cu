#include "kernels_host.h"
#include "kernels_device.cuh"
#include "morton_code.cuh"

#include <cuda_runtime.h>

#include <thrust/sort.h>

using namespace lbvh;

void morton_codes_host(unsigned long long int* h_mortonCodes, float* h_points, int num_points)
{
    int size_points = num_points * 3 * sizeof(float);
    int size_morton = num_points * sizeof(unsigned long long int);

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_points + threadsPerBlock - 1) / threadsPerBlock;

    float* d_points;
    cudaMalloc(&d_points, size_points);
    cudaMemcpy(d_points, h_points, size_points, cudaMemcpyHostToDevice);

    // Get the extent of the point cloud
    AABB extent = getExtent(h_points, num_points); 

    unsigned long long int* d_mortonCodes;
    cudaMalloc(&d_mortonCodes, size_morton);
    cudaMemcpy(d_mortonCodes, h_mortonCodes, size_morton, cudaMemcpyHostToDevice);
    morton_code_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_mortonCodes, d_points, num_points, extent);

    cudaDeviceSynchronize();

    cudaMemcpy(h_mortonCodes, d_mortonCodes, size_morton, cudaMemcpyDeviceToHost);

    cudaFree(d_points);
    cudaFree(d_mortonCodes);

    return;

}

void radix_sort(unsigned long long int* keys, int* values, size_t num_points)
{
    thrust::sort_by_key(keys, keys + num_points, values);
}