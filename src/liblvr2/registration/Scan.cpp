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

#include <iostream>
#include <fstream>
#include <stdexcept>

#include <lvr2/io/IOUtils.hpp>
#include <lvr2/registration/TreeUtils.hpp>

using namespace std;

namespace lvr2
{

Scan::Scan(PointBufferPtr points, const Matrix4f& pose)
    : m_pose(pose), m_initialPose(pose), m_deltaPose(Matrix4f::Identity()), m_transformChanged(false), m_transformChange(Matrix4f::Identity())
{
    size_t n = points->numPoints();
    m_points.resize(n);

    floatArr src = points->getPointArray();

    #pragma omp parallel for
    for (size_t i = 0; i < n; i++)
    {
        Eigen::Vector4f extended(src[i * 3], src[i * 3 + 1], src[i * 3 + 2], 1.0);
        m_points[i] = (pose * extended).block<3, 1>(0, 0);
    }
}

void Scan::transform(const Matrix4f& transform, bool writeFrame, ScanUse use)
{
    m_pose = transform * m_pose;
    m_deltaPose = transform * m_deltaPose;

    m_transformChanged = true;
    m_transformChange = transform * m_transformChange;

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
    Vector3f origin = getPosition();

    size_t i = 0;
    while (i < count())
    {
        if ((m_points[i] - origin).squaredNorm() <= sqDist)
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
    Vector3f origin = getPosition();

    size_t i = 0;
    while (i < count())
    {
        if ((m_points[i] - origin).squaredNorm() >= sqDist)
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
    if (m_transformChanged)
    {
        const Eigen::Vector3f& p = m_points[index];
        Eigen::Vector4f extended(p.x(), p.y(), p.z(), 1.0);
        return (m_transformChange * extended).block<3, 1>(0, 0);
    }
    return m_points[index];
}

Vector3f* Scan::points()
{
    if (m_transformChanged)
    {
        #pragma omp parallel for
        for (int i = 0; i < m_points.size(); i++)
        {
            const Eigen::Vector3f& p = m_points[i];
            Eigen::Vector4f extended(p.x(), p.y(), p.z(), 1.0);
            m_points[i] = (m_transformChange * extended).block<3, 1>(0, 0);
        }
        m_transformChange = Matrix4f::Identity();
        m_transformChanged = false;
    }
    return m_points.data();
}

size_t Scan::count() const
{
    return m_points.size();
}

const Matrix4f& Scan::getPose() const
{
    return m_pose;
}

const Matrix4f& Scan::getDeltaPose() const
{
    return m_deltaPose;
}

const Matrix4f& Scan::getInitialPose() const
{
    return m_initialPose;
}

Vector3f Scan::getPosition() const
{
    return m_pose.block<3, 1>(0, 3);
}

void Scan::addFrame(ScanUse use)
{
    m_frames.push_back(make_pair(m_pose, use));
}

void Scan::writeFrames(std::string path) const
{
    ofstream out(path);
    for (const std::pair<Matrix4f, ScanUse>& frame : m_frames)
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

    #pragma omp parallel for
    for (int i = 0; i < count(); i++)
    {
        arr[3 * i + 0] = m_points[i].x();
        arr[3 * i + 1] = m_points[i].y();
        arr[3 * i + 2] = m_points[i].z();
    }

    ret->setPointArray(arr, count());

    return ret;
}

void Scan::addScanToMeta(ScanPtr scan)
{
    m_points.reserve(scan->count());
    m_points.insert(m_points.end(), scan->m_points.begin(), scan->m_points.end());
    m_deltaPose = scan->getDeltaPose();
}

} /* namespace lvr2 */
