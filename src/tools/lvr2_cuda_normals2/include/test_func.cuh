#ifndef TEST_FUNC_CUH
#define TEST_FUNC_CUH


// Working:
#include <cuda_runtime.h>
#include "vec_math.h"
#include <vector_functions.h>
#include <vector_types.h>
#include <cuda/std/limits>
#include <cuda/std/cmath>
#include "aabb.cuh"
#include "morton_code.cuh"
#include "lbvh.cuh"
#include "lbvh_kernels.cuh"
#include "static_priorityqueue.cuh"
#include "query.cuh"
#include "query_knn.cuh"

// Failing:
// #include <cstdlib>   // Needed?
// #include <cmath>     // Needed?

// #include "kernels_host.h"

// #include <thrust/sort.h>
// #include <string>
// #include <iostream>
// #include <fstream>
// #include <vector>


namespace lbvh
{
__device__ void test_func()
{
   float3 test;
   test.x = threadIdx.x;

   AABB abab;

   printf("Test: %f\n", test.x);
}
}
#endif // TEST_FUNC_CUH