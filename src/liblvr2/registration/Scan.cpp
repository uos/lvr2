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

Scan::Scan(PointBufferPtr points, const Matrix4d& pose)
    : m_pose(pose), m_initialPose(pose), m_deltaPose(Matrix4d::Identity())
{
    m_count = points->numPoints();
    m_points = boost::shared_array<Vector3d>(new Vector3d[m_count]);

    floatArr src = points->getPointArray();

    #pragma omp parallel for
    for (size_t i = 0; i < m_count; i++)
    {
        m_points[i] = Vector3d(src[i * 3], src[i * 3 + 1], src[i * 3 + 2]);
    }
}

Scan::Scan()
    : m_count(0), m_pose(Matrix4d::Identity()), m_initialPose(Matrix4d::Identity()), m_deltaPose(Matrix4d::Identity())
{ }

void Scan::transform(const Matrix4d& transform, bool writeFrame)
{
    m_pose = transformRegistration(transform, m_pose);
    m_deltaPose = transformRegistration(transform, m_deltaPose);

    if (writeFrame)
    {
        addFrame(ScanUse::UPDATED);
    }
}

void Scan::reduce(double voxelSize)
{
    m_count = octreeReduce(m_points.get(), m_count, voxelSize);
}

void Scan::setMinDistance(double minDistance)
{
    double sqDist = minDistance * minDistance;

    size_t i = 0;
    while (i < m_count)
    {
        if (m_points[i].squaredNorm() <= sqDist)
        {
            m_count--;
            std::swap(m_points[i], m_points[m_count]);
        }
        else
        {
            i++;
        }
    }
}
void Scan::setMaxDistance(double maxDistance)
{
    double sqDist = maxDistance * maxDistance;

    size_t i = 0;
    while (i < m_count)
    {
        if (m_points[i].squaredNorm() >= sqDist)
        {
            m_count--;
            std::swap(m_points[i], m_points[m_count]);
        }
        else
        {
            i++;
        }
    }
}

const Vector3d& Scan::getPoint(size_t index) const
{
    if (index >= m_count)
    {
        throw out_of_range("getPoint on Scan out of Range");
    }
    return m_points[index];
}
Vector3d Scan::getPointTransformed(size_t index) const
{
    Eigen::Vector4d extended;
    extended << getPoint(index), 1.0;
    return (m_pose * extended).block<3, 1>(0, 0);
}
size_t Scan::count() const
{
    return m_count;
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

void Scan::addFrame(ScanUse use)
{
    m_frames.push_back(make_pair(m_pose, use));
}

void Scan::writeFrames(std::string path) const
{
    ofstream out(path);
    for (const std::pair<Matrix4d, ScanUse>& frame : m_frames)
    {
        for(int i = 0; i < 16; i++)
        {
            out << frame.first(i) << " ";
        }

        out << (int)frame.second << endl;
    }
}

} /* namespace lvr2 */
