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

/**
 * Scan.hpp
 *
 *  @date May 6, 2019
 *  @author Malte Hillmann
 */
#ifndef SCAN_HPP_
#define SCAN_HPP_

#include <Eigen/Dense>
#include <lvr2/io/PointBuffer.hpp>
using Eigen::Matrix4d;
using Eigen::Vector3d;

namespace lvr2
{

enum class ScanUse
{
    INVALID = 0,
    UPDATED = 1,
    UNUSED = 2,
};

class Scan
{
public:
    using PointArray = boost::shared_array<Vector3d>;

    Scan(PointBufferPtr points, const Matrix4d& pose);

    void transform(const Matrix4d& transform, bool writeFrame = true);
    void addFrame(ScanUse use = ScanUse::UNUSED);

    void reduce(double voxelSize);
    void setMinDistance(double minDistance);
    void setMaxDistance(double maxDistance);

    virtual const Vector3d& getPoint(size_t index) const;
    virtual Vector3d getPointTransformed(size_t index) const;
    size_t count() const;

    const Matrix4d& getPose() const;
    const Matrix4d& getDeltaPose() const;
    const Matrix4d& getInitialPose() const;

    void writeFrames(std::string path) const;

protected:
    Scan();

    PointArray  m_points;
    size_t      m_count;

    Matrix4d    m_pose;
    Matrix4d    m_deltaPose;
    Matrix4d    m_initialPose;

    std::vector<std::pair<Matrix4d, ScanUse>> m_frames;
};

using ScanPtr = std::shared_ptr<Scan>;

} /* namespace lvr2 */

#endif /* SCAN_HPP_ */
