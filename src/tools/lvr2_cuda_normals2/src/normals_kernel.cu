#include "normals_kernel.cuh"

#include <stdio.h>

using namespace lbvh;

__global__ void lbvh::calculate_normals_kernel(float* points, size_t num_normals, 
    unsigned int* n_neighbors_out, unsigned int* indices_out, 
    unsigned int* neigh_sum,
    float* normals)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid == 0)
    {   
        printf("Checking kernel... \n");
        printf("%f \n", points[0]);
        printf("%d \n", num_normals);
        printf("%d \n", n_neighbors_out[0]);
        printf("%d \n", indices_out[0]);
        printf("%d \n", neigh_sum[0]);
        printf("%f \n", normals[0]);
        printf("Kernel working fine \n");
    }
}