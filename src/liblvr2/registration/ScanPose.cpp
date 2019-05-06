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
 * ScanPose.cpp
 *
 *  @date May 6, 2019
 *  @author Malte Hillmann
 */
#include <lvr2/registration/ScanPose.hpp>

#include <iostream>
#include <fstream>

using namespace std;

const double DEG_TO_RAD = M_PI / 180.0;

namespace lvr2
{

ScanPose::ScanPose()
    : m_position(Vector3d::Zero()), m_angles(Vector3d::Zero())
{ }

ScanPose::ScanPose(const double* pose, bool radians)
    : m_position(pose[0], pose[1], pose[2]), m_angles(pose[3], pose[4], pose[5])
{
    if (!radians)
    {
        m_angles *= DEG_TO_RAD;
    }
}

ScanPose::ScanPose(const float* pose, bool radians)
    : m_position(pose[0], pose[1], pose[2]), m_angles(pose[3], pose[4], pose[5])
{
    if (!radians)
    {
        m_angles *= DEG_TO_RAD;
    }
}

ScanPose::ScanPose(double x, double y, double z, double r, double t, double s, bool radians)
    : m_position(x, y, z), m_angles(r, t, s)
{
    if (!radians)
    {
        m_angles *= DEG_TO_RAD;
    }
}

ScanPose::ScanPose(const Matrix4d& pose)
{
    // Calculate Y-axis angle
    if (pose(0, 0) > 0.0)
    {
        m_angles.y() = asin(pose(2, 0));
    }
    else
    {
        m_angles.y() = M_PI - asin(pose(2, 0));
    }

    double C = cos(m_angles.y());
    if (fabs(C) < 0.005) // Gimbal lock?
    {
        // Gimbal lock has occurred
        m_angles.x() = 0.0;
        m_angles.z() = atan2(pose(0, 1), pose(1, 1));
    }
    else
    {
        m_angles.x() = atan2(-pose(2, 1) / C, pose(2, 2) / C);
        m_angles.z() = atan2(-pose(1, 0) / C, pose(0, 0) / C);
    }

    m_position = pose.block<3, 1>(0, 3);
}

ScanPose::ScanPose(const Vector3d& position, const Vector3d& angles, bool radians)
    : m_position(position), m_angles(angles)
{
    if (!radians)
    {
        m_angles *= DEG_TO_RAD;
    }
}

ScanPose ScanPose::fromFile(const string& file)
{
    ScanPose pose;
    ifstream in(file.c_str());
    if(in.good())
    {
        in >> pose.m_position.x() >> pose.m_position.y() >> pose.m_position.z();
        in >> pose.m_angles.x() >> pose.m_angles.y() >> pose.m_angles.z();
        pose.m_angles *= DEG_TO_RAD;
    }
    else
    {
        cerr << "Unable to open " << file << endl;
    }
    return pose;
}

Matrix4d ScanPose::toMatrix() const
{
    Eigen::Matrix4d mat = Eigen::Matrix4d::Identity();
    mat.block<3, 3>(0, 0) = Eigen::AngleAxisd(m_angles.x(), Eigen::Vector3d::UnitX()).matrix()
                            * Eigen::AngleAxisd(m_angles.y(), Eigen::Vector3d::UnitY())
                            * Eigen::AngleAxisd(m_angles.z(), Eigen::Vector3d::UnitZ());

    mat.block<3, 1>(0, 3) = m_position;
    return mat;
}

std::pair<Vector3d, Vector3d> ScanPose::toPosAngle(bool radians) const
{
    if (radians)
    {
        return make_pair(m_position, m_angles);
    }
    else
    {
        return make_pair(m_position, m_angles / DEG_TO_RAD);
    }
}

std::array<double, 6> ScanPose::toArray(bool radians) const
{
    Vector3d angles = m_angles;
    if (!radians)
    {
        angles /= DEG_TO_RAD;
    }

    return {
        m_position[0], m_position[1], m_position[2],
        angles[0], angles[1], angles[2],
    };
}

std::array<float, 6> ScanPose::toFloatArray(bool radians) const
{
    Eigen::Vector3f floatPos = m_position.cast<float>();
    Eigen::Vector3f floatAngles = m_angles.cast<float>();

    if (!radians)
    {
        floatAngles /= DEG_TO_RAD;
    }

    return {
        floatPos[0], floatPos[1], floatPos[2],
        floatAngles[0], floatAngles[1], floatAngles[2],
    };
}

} /* namespace lvr2 */
