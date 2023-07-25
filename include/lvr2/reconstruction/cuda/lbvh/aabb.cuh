/**
 * Copyright (c) 2023, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once
#ifndef LVR2_AABB_CUH
#define LVR2_AABB_CUH

//#include <cuda_runtime.h>

namespace lvr2
{

namespace lbvh 
{
    struct AABB
    {
        float3 min;
        float3 max;
    };

    /**
     * @brief Returns the centroid of an axis-aligned bounding box
     * 
     * @param box The bounding box
     * 
     * @return The centroid
     */
    __device__
    inline float3 centroid(const AABB& box) noexcept
    {
        float3 c;
        c.x = (box.max.x + box.min.x) * 0.5;
        c.y = (box.max.y + box.min.y) * 0.5;
        c.z = (box.max.z + box.min.z) * 0.5;
        return c;
    }

    /**
     * @brief Merges two bounding boxes 
     * 
     * @param lhs First bounding box
     * @param rhs Second bounding box
     * 
     * @return Merged bounding box
     */
    __device__
    inline AABB merge(const AABB& lhs, const AABB& rhs) noexcept
    {
        AABB merged;
        merged.max.x = fmaxf(lhs.max.x, rhs.max.x);
        merged.max.y = fmaxf(lhs.max.y, rhs.max.y);
        merged.max.z = fmaxf(lhs.max.z, rhs.max.z);
        merged.min.x = fminf(lhs.min.x, rhs.min.x);
        merged.min.y = fminf(lhs.min.y, rhs.min.y);
        merged.min.z = fminf(lhs.min.z, rhs.min.z);
        return merged;
    }


    /**
     * @brief Returns max of 3 float values
     * 
     * @param first first float value
     * @param second second float value
     * @param thrid third float value
     * 
     * @return Max of the 3 float values
     */
    __forceinline__ __device__ float fmax3(float first, float second, float third) noexcept
    {
        return fmaxf(first, fmaxf(second, third));
    }

    /**
     * @brief Calculates the squared length of a vector
     * 
     * @param vec the vector
     * 
     * @return The squared length of vec 
     */
    __forceinline__ __device__ float sq_length3(const float3& vec) noexcept {
        return vec.x * vec.x + vec.y * vec.y + vec.z * vec.z;
    }

    /**
     * @brief  Calculates the distance of a point to a bounding box
     * 
     * @param p the point
     * @param aabb the bounding box
     * 
     * @return Squared distance of p to aabb
     */
    __forceinline__ __device__
    float dist_2_aabb(const float3& p, const AABB& aabb) noexcept
    {
        float sqDist(0);
        float v;

        if (p.x < aabb.min.x) v = aabb.min.x;
        if (p.x > aabb.max.x) v = aabb.max.x;
        if (p.x < aabb.min.x || p.x > aabb.max.x) sqDist += (v-p.x) * (v-p.x);

        if (p.y < aabb.min.y) v = aabb.min.y;
        if (p.y > aabb.max.y) v = aabb.max.y;
        if (p.y < aabb.min.y || p.y > aabb.max.y) sqDist += (v-p.y) * (v-p.y);

        if (p.z < aabb.min.z) v = aabb.min.z;
        if (p.z > aabb.max.z) v = aabb.max.z;
        if (p.z < aabb.min.z || p.z > aabb.max.z) sqDist += (v-p.z) * (v-p.z);
        return sqDist;
    }
}

} // namespace lvr2

#endif // LVR2_AABB_CUH