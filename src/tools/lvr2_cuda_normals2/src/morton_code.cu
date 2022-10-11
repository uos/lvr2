#include "morton_code.cuh"
#include "aabb.cuh"

#include <stdio.h>

using namespace lbvh;

// Get the extent of the points 
// (minimum and maximum values in each dimension)
__device__ __host__ AABB lbvh::getExtent(float* points, size_t num_points)
{
    float min_x = INT_MAX;
    float min_y = INT_MAX;
    float min_z = INT_MAX;

    float max_x = INT_MIN; 
    float max_y = INT_MIN; 
    float max_z = INT_MIN;

    for(int i = 0; i < 3 * num_points; i += 3)
    {
        if(points[i + 0] < min_x)
        {
            min_x = points[i + 0];
        }

        if(points[i + 1] < min_y)
        {
            min_y = points[i + 1];
        }

        if(points[i + 2] < min_z)
        {
            min_z = points[i + 2];
        }

        if(points[i + 0] > max_x)
        {
            max_x = points[i + 0];
        }

        if(points[i + 1] > max_y)
        {
            max_y = points[i + 1];
        }

        if(points[i + 2] > max_z)
        {
            max_z = points[i + 2];
        }
    }
    
    AABB extent;
    extent.min.x = min_x;
    extent.min.y = min_y;
    extent.min.z = min_z;
    
    extent.max.x = max_x;
    extent.max.y = max_y;
    extent.max.z = max_z;
    
    return extent;
}

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
