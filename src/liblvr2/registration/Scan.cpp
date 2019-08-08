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
 * Scan.cpp
 *
 *  @date May 6, 2019
 *  @author Malte Hillmann
 */
#include <lvr2/registration/Scan.hpp>

#include <lvr2/registration/TreeUtils.hpp>

#include <fstream>

using namespace std;

namespace lvr2
{

Scan::Scan(PointBufferPtr points, const Matrix4d& pose)
    : m_pose(pose), m_deltaPose(Matrix4d::Identity()), m_initialPose(pose)
{
    size_t n = points->numPoints();
    m_points.resize(n);

    floatArr src = points->getPointArray();

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++)
    {
        m_points[i] = Vector3f(src[i * 3], src[i * 3 + 1], src[i * 3 + 2]);
    }
}

void Scan::transform(const Matrix4d& transform, bool writeFrame, ScanUse use)
{
    m_pose = transform * m_pose;
    m_deltaPose = transform * m_deltaPose;

    if (writeFrame)
    {
        addFrame(use);
    }
}

void Scan::reduce(float voxelSize)
{
    size_t count = octreeReduce(m_points.data(), m_points.size(), voxelSize);
    m_points.resize(count);
    m_points.shrink_to_fit();
}

void Scan::setMinDistance(float minDistance)
{
    float sqDist = minDistance * minDistance;

    size_t i = 0;
    while (i < count())
    {
        if (m_points[i].squaredNorm() <= sqDist)
        {
            if (i < count() - 1)
            {
                m_points[i] = m_points.back();
            }
            m_points.pop_back();
        }
        else
        {
            i++;
        }
    }
    m_points.shrink_to_fit();
}
void Scan::setMaxDistance(float maxDistance)
{
    float sqDist = maxDistance * maxDistance;

    size_t i = 0;
    while (i < count())
    {
        if (m_points[i].squaredNorm() >= sqDist)
        {
            if (i < count() - 1)
            {
                m_points[i] = m_points.back();
            }
            m_points.pop_back();
        }
        else
        {
            i++;
        }
    }
    m_points.shrink_to_fit();
}

Vector3f Scan::getPoint(size_t index) const
{
    const Vector3f& p = m_points[index];
    Eigen::Vector4d extended(p.x(), p.y(), p.z(), 1.0);
    return (m_pose * extended).block<3, 1>(0, 0).cast<float>();
}

size_t Scan::count() const
{
    return m_points.size();
}

const Matrix4d& Scan::getPose() const
{
    return m_pose;
}

const Matrix4d& Scan::getDeltaPose() const
{
    return m_deltaPose;
}

const Matrix4d& Scan::getInitialPose() const
{
    return m_initialPose;
}

Vector3f Scan::getPosition() const
{
    return m_pose.block<3, 1>(0, 3).cast<float>();
}

void Scan::addFrame(ScanUse use)
{
    m_frames.push_back(make_pair(m_pose, use));
}

void Scan::writeFrames(std::string path) const
{
    ofstream out(path);
    for (const std::pair<Matrix4d, ScanUse>& frame : m_frames)
    {
        for (int i = 0; i < 16; i++)
        {
            out << frame.first(i) << " ";
        }

        out << (int)frame.second << endl;
    }
}

PointBufferPtr Scan::toPointBuffer() const
{
    auto ret = make_shared<PointBuffer>();
    auto arr = floatArr(new float[3 * count()]);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < count(); i++)
    {
        arr[3 * i + 0] = m_points[i].x();
        arr[3 * i + 1] = m_points[i].y();
        arr[3 * i + 2] = m_points[i].z();
    }

    ret->setPointArray(arr, count());

    return ret;
}

} /* namespace lvr2 */
