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

#include <lvr2/io/IOUtils.hpp>

using namespace std;

namespace lvr2
{

Scan::Scan(PointBufferPtr points, const Matrix4d& pose)
    : m_points(points), m_pose(pose), m_initialPose(pose), m_deltaPose(Matrix4d::Identity())
{ }

void Scan::transform(const Matrix4d& transform, bool writeFrame)
{
    m_pose = transformRegistration(m_pose, transform);
    m_deltaPose = transformRegistration(m_deltaPose, transform);

    if (writeFrame)
    {
        addFrame(ScanUse::UPDATED);
    }
}

const PointBufferPtr& Scan::getPoints() const
{
    return m_points;
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
