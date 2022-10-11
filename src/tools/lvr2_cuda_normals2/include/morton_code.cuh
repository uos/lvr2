#pragma once
#ifndef MORTON_CODE_CUH
#define MORTON_CODE_CUH

#include "aabb.cuh"

namespace lbvh
{

typedef unsigned long long int HashType;

// Get the extent of the points 
// (minimum and maximum values in each dimension)
__device__ __host__ AABB getExtent(float* d_points, size_t num_points);


__device__ HashType expand_bits(HashType v);

// Calculates a Morton code for the
// given 3D point located within the unit cube [0,1].
__device__ HashType morton_code(float3 xyz, float resolution = 1024.0f) noexcept;

// Returns the highest differing bit of the two morton codes
__device__ int highest_bit(HashType lhs, HashType rhs) noexcept;

}
#endif // MORTON_CODE_CUH