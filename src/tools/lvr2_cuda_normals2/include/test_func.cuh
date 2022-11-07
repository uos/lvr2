#ifndef TEST_FUNC_CUH
#define TEST_FUNC_CUH


// Working:
#include <cuda_runtime.h>
#include "vec_math.h"
#include <vector_functions.h>
#include <vector_types.h>

// Failing:
// #include <cstdlib>   // Needed?
// #include <cmath>     // Needed?
// #include <cuda.h>
// #include <nvrtc.h>
// #include <thrust/sort.h>
// #include <string>
// #include <iostream>
// #include <fstream>
// #include <vector>
// #include <stdio.h>
// #include <cuda/std/cmath>
// #include <cuda/std/limits>


// #include "aabb.cuh"
// #include "kernels_host.h"
// #include "static_priorityqueue.cuh"
// #include "query.cuh"
// #include "GPUErrorCheck.h"
// #include "morton_code.cuh"
// #include "lbvh.cuh"
// #include "aabb.cuh"
// #include "lbvh_kernels.cuh"

namespace lbvh
{
__device__ void test_func()
{
   float3 test;
   test.x = threadIdx.x;

   printf("Test: %f\n", test.x);
}
}
#endif // TEST_FUNC_CUH