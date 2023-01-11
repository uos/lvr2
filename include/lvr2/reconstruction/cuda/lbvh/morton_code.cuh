#pragma once
#ifndef LVR2_MORTON_CODE_CUH
#define LVR2_MORTON_CODE_CUH

#include "aabb.cuh"

namespace lvr2
{

namespace lbvh
{

typedef unsigned long long int HashType;

__device__ HashType expand_bits(HashType v);

// Calculates a Morton code for the
// given 3D point located within the unit cube [0,1].
__device__ HashType morton_code(
    float3 xyz, 
    float resolution = 1024.0f
) noexcept;

// Returns the highest differing bit of the two morton codes
__device__ int highest_bit(HashType lhs, HashType rhs) noexcept;

} // namespace lbvh

} // namespace lvr2

#endif // LVR2_MORTON_CODE_CUH