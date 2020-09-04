/**
 * Copyright (c) 2020, University Osnabrück
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

/*
 * EmbreeRaycaster.hpp
 *
 *  @date 25.01.2020
 *  @author Alexander Mock <amock@uos.de>
 */

#ifndef LVR2_ALGORITHM_RAYCASTING_EMBREERAYCASTER
#define LVR2_ALGORITHM_RAYCASTING_EMBREERAYCASTER

#include <embree3/rtcore.h>
#include <stdio.h>

#include "lvr2/algorithm/raycasting/RaycasterBase.hpp"
#include "lvr2/io/MeshBuffer.hpp"
#include "lvr2/types/MatrixTypes.hpp"
#include "Intersection.hpp"

namespace lvr2 {

void EmbreeErrorFunction(void* userPtr, enum RTCError error, const char* str);

template<typename IntT>
class EmbreeRaycaster : public RaycasterBase<IntT> {
public:
    EmbreeRaycaster(const MeshBufferPtr mesh);
    ~EmbreeRaycaster();

    /**
     * @brief Cast a single ray onto the mesh
     * 
     * @param[in] origin Ray origin 
     * @param[in] direction Ray direction
     * @param[out] intersection User defined intersection output 
     * @return true  Intersection found
     * @return false  Not intersection found
     */
    bool castRay(
        const Vector3f& origin,
        const Vector3f& direction,
        IntT& intersection);

protected:

    RTCDevice initializeDevice();
    RTCScene initializeScene(RTCDevice device, const MeshBufferPtr mesh);

    inline RTCRayHit lvr2embree(
        const Vector3f& origin, 
        const Vector3f& direction) const
    {
        RTCRayHit rayhit;
        rayhit.ray.org_x = origin.x();
        rayhit.ray.org_y = origin.y();
        rayhit.ray.org_z = origin.z();
        rayhit.ray.dir_x = direction.x();
        rayhit.ray.dir_y = direction.y();
        rayhit.ray.dir_z = direction.z();
        rayhit.ray.tnear = 0;
        rayhit.ray.tfar = INFINITY;
        rayhit.ray.mask = 0;
        rayhit.ray.flags = 0;
        rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
        rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
        return rayhit;
    }

    RTCDevice m_device;
    RTCScene m_scene;
    RTCIntersectContext m_context;
};

} // namespace lvr2

#include "EmbreeRaycaster.tcc"

#endif // LVR2_ALGORITHM_RAYCASTING_EMBREERAYCASTER