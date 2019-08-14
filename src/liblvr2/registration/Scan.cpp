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

Scan::Scan(PointBufferPtr points, const Transformd& pose)
    : m_pose(pose), m_deltaPose(Transformd::Identity()), m_initialPose(pose)
{
    if (points)
    {
        m_points = points->getPointArray();
        m_numPoints = points->numPoints();
    }
    else
    {
        m_numPoints = 0;
    }
}

void Scan::transform(const Transformd& transform, bool writeFrame, ScanUse use)
{
    m_pose = transform * m_pose;
    m_deltaPose = transform * m_deltaPose;

    if (writeFrame)
    {
        addFrame(use);
    }
}

void Scan::reduce(double voxelSize, int maxLeafSize)
{
    m_numPoints = octreeReduce((Eigen::Vector3f*)m_points.get(), m_numPoints, voxelSize, maxLeafSize);
}

void Scan::setMinDistance(double minDistance)
{
    double sqDist = minDistance * minDistance;

    float* cur = m_points.get();
    float* end = m_points.get() + m_numPoints * 3;
    while (cur < end)
    {
        if (pow(*cur, 2.0) + pow(*(cur + 1), 2.0) + pow(*(cur + 2), 2.0) <= sqDist)
        {
            end -= 3;
            if (cur < end)
            {
                copy_n(end, 3, cur);
            }
            m_numPoints--;
        }
        else
        {
            cur += 3;
        }
    }
}

void Scan::setMaxDistance(double maxDistance)
{
    double sqDist = maxDistance * maxDistance;

    float* cur = m_points.get();
    float* end = m_points.get() + m_numPoints * 3;
    while (cur < end)
    {
        if (pow(*cur, 2.0) + pow(*(cur + 1), 2.0) + pow(*(cur + 2), 2.0) <= sqDist)
        {
            end -= 3;
            if (cur < end)
            {
                copy_n(end, 3, cur);
            }
            m_numPoints--;
        }
        else
        {
            cur += 3;
        }
    }
}

void Scan::trim()
{
    floatArr array = floatArr(new float[m_numPoints * 3]);
    copy_n(m_points.get(), m_numPoints * 3, array.get());
    m_points.swap(array);
}

Vector3d Scan::getPoint(size_t index) const
{
    float* p = m_points.get() + index * 3;
    Eigen::Vector4d extended(*p, *(p + 1), *(p + 2), 1.0);
    return (m_pose * extended).block<3, 1>(0, 0);
}

size_t Scan::numPoints() const
{
    return m_numPoints;
}

const Transformd& Scan::getPose() const
{
    return m_pose;
}

const Transformd& Scan::getDeltaPose() const
{
    return m_deltaPose;
}

const Transformd& Scan::getInitialPose() const
{
    return m_initialPose;
}

Vector3d Scan::getPosition() const
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
    for (const std::pair<Transformd, ScanUse>& frame : m_frames)
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
    ret->setPointArray(m_points, m_numPoints);
    return ret;
}

Vector3fArr Scan::toVector3fArr() const
{
    auto ret = Vector3fArr(new Eigen::Vector3f[m_numPoints]);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < m_numPoints; i++)
    {
        ret[i] = getPoint(i).cast<float>();
    }

    return ret;
}

} /* namespace lvr2 */
