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

using namespace std;

namespace lvr2
{

Scan::Scan(PointBufferPtr points, const Matrix4d& pose)
    : m_points(points), m_pose(pose), m_deltaPose(Matrix4d::Identity())
{
    m_frames.push_back(m_pose);
}

void Scan::transform(const Matrix4d& transform)
{
    m_pose *= transform;
    m_deltaPose *= transform;
    addFrame();
}

const PointBufferPtr& Scan::getPoints() const
{
    return m_points;
}
const Matrix4d& Scan::getPose() const
{
    return m_pose;
}

void Scan::addFrame()
{
    m_frames.push_back(m_pose);
}

void Scan::writeFrames(std::string path) const
{
    ofstream out(path);
    for (const Matrix4d& frame : m_frames)
    {
        for(int i = 0; i < 16; i++)
        {
            out << frame(i) << " ";
        }

        out << "1" << endl;
    }
}

} /* namespace lvr2 */
