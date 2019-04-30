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

#include <lvr2/io/MeshBuffer.hpp>
#include <lvr2/geometry/BaseVector.hpp>
#include <lvr2/geometry/Normal.hpp>

namespace lvr2
{

/**
 * @brief RaycasterBase interface
 */
template <typename PointT, typename NormalT>
class RaycasterBase {
public:

    /**
     * @brief Constructor: Stores mesh as member
     */
    RaycasterBase(const MeshBufferPtr mesh);

    virtual bool castRay(
        const PointT& origin,
        const NormalT& direction,
        PointT& intersection
    ) = 0;

    virtual void castRays(
        const PointT& origin,
        const std::vector<NormalT >& directions,
        std::vector<PointT >& intersections,
        std::vector<uint8_t>& hits
    ) = 0;

    virtual void castRays(
        const std::vector<PointT >& origins,
        const std::vector<NormalT >& directions,
        std::vector<PointT >& intersections,
        std::vector<uint8_t>& hits
    ) = 0;

private:
    const MeshBufferPtr m_mesh;
};

} // namespace lvr2

#include <lvr2/algorithm/raycasting/RaycasterBase.tcc>