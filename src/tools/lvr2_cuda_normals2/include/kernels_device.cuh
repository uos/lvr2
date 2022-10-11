#ifndef KERNELS_DEVICE_CUH
#define KERNELS_DEVICE_CUH

#include <cuda_runtime.h>
#include "aabb.cuh"

namespace lbvh
{

__global__
void morton_code_kernel(unsigned long long int* d_mortonCodes, 
                    float* d_points,
                    int num_points, 
                    AABB extent);

}
#endif // KERNELS_DEVICE_CUH