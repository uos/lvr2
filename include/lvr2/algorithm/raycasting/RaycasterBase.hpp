/**
 * Copyright (c) 2018, University Osnabrück
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
 * RaycasterBase.hpp
 *
 *  @date 25.01.2019
 *  @author Alexander Mock <amock@uos.de>
 */

#pragma once
#ifndef LVR_RAYCASTER_BASE_HPP
#define LVR_RAYCASTER_BASE_HPP

#include <memory>
#include "lvr2/io/MeshBuffer.hpp"
#include "lvr2/types/MatrixTypes.hpp"

#include "Intersection.hpp"

namespace lvr2
{

/**
 * @brief RaycasterBase interface
 */

template<typename IntT>
class RaycasterBase {
public:
    /**
     * @brief Constructor: Stores mesh as member
     */
    RaycasterBase(const MeshBufferPtr mesh);

    // PURE VIRTUAL
    virtual bool castRay(
        const Vector3f& origin,
        const Vector3f& direction,
        IntT& intersection
    ) = 0;
    
    // VIRTUAL WITH DEFAULTS
    /**
     * Cast Ray. one origin. multiple directions (vector form)
     */
    virtual void castRays(
        const Vector3f& origin,
        const std::vector<Vector3f>& directions,
        std::vector<IntT>& intersections,
        std::vector<uint8_t>& hits
    );

    /**
     * Cast Ray. one origin. multiple directions (matrix form)
     */
    virtual void castRays(
        const Vector3f& origin,
        const std::vector<std::vector<Vector3f> >& directions,
        std::vector< std::vector<IntT> >& intersections,
        std::vector< std::vector<uint8_t> >& hits
    );

    /**
     * Cast Ray. pair of origin and direction
     */
    virtual void castRays(
        const std::vector<Vector3f>& origins,
        const std::vector<Vector3f>& directions,
        std::vector<IntT>& intersections,
        std::vector<uint8_t>& hits
    );

    /**
     * Cast Ray. multiple origins. each origins can have multiple directions.
     */
    virtual void castRays(
        const std::vector<Vector3f>& origins,
        const std::vector<std::vector<Vector3f> >& directions,
        std::vector<std::vector<IntT> >& intersections,
        std::vector<std::vector<uint8_t> >& hits
    );

private:
    const MeshBufferPtr m_mesh;
};

template<typename IntT>
using RaycasterBasePtr = std::shared_ptr<RaycasterBase<IntT> >;

} // namespace lvr2

#include "RaycasterBase.tcc"

#endif // LVR_RAYCASTER_BASE_HPP