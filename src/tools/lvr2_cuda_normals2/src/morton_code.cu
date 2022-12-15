#include "morton_code.cuh"

#include <stdio.h>

using namespace lbvh;


__device__ HashType lbvh::expand_bits(HashType v)
{
    v = (v * 0x000100000001u) & 0xFFFF00000000FFFFu;
    v = (v * 0x000000010001u) & 0x00FF0000FF0000FFu;
    v = (v * 0x000000000101u) & 0xF00F00F00F00F00Fu;
    v = (v * 0x000000000011u) & 0x30C30C30C30C30C3u;
    v = (v * 0x000000000005u) & 0x9249249249249249u;
    return v;
}


// Calculates a Morton code for the
// given 3D point located within the unit cube [0,1].
__device__ HashType lbvh::morton_code(float3 xyz, float resolution) noexcept
{
    resolution *= resolution; // increase the resolution for 64 bit codes

    xyz.x = ::fminf(::fmaxf(xyz.x * resolution, 0.0f), resolution - 1.0f);
    xyz.y = ::fminf(::fmaxf(xyz.y * resolution, 0.0f), resolution - 1.0f);
    xyz.z = ::fminf(::fmaxf(xyz.z * resolution, 0.0f), resolution - 1.0f);
    const HashType xx = expand_bits(static_cast<HashType>(xyz.x));
    const HashType yy = expand_bits(static_cast<HashType>(xyz.y));
    const HashType zz = expand_bits(static_cast<HashType>(xyz.z));
    return xx * 4 + yy * 2 + zz;
}

// Returns the highest differing bit of the two morton codes
__device__ int lbvh::highest_bit(HashType lhs, HashType rhs) noexcept
{
    return lhs ^ rhs;
}
