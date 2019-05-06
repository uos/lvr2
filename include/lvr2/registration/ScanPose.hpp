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
 * Pose.hpp
 *
 *  @date May 6, 2019
 *  @author Malte Hillmann
 */
#ifndef SCANPOSE_HPP_
#define SCANPOSE_HPP_

#include <utility>
#include <array>
#include <string>
#include <Eigen/Dense>
using Eigen::Matrix4d;
using Eigen::Vector3d;

namespace lvr2
{

class ScanPose
{
public:
    /**
     * @brief Creates a Pose set to 0
     */
    ScanPose();

    /**
     * @brief Creates a Pose from a Pose array containing { x, y, z, r, t, s }
     *
     * Does not take ownership of the array.
     *
     * @param pose The Pose array
     * @param radians true = rotation is in Radians, false = rotation is in Degrees
     */
    ScanPose(const double* pose, bool radians = true);

    /**
     * @brief Creates a Pose from a Pose array containing { x, y, z, r, t, s }
     *
     * Does not take ownership of the array.
     *
     * @param pose The Pose array
     * @param radians true = rotation is in Radians, false = rotation is in Degrees
     */
    ScanPose(const float* pose, bool radians = true);

    /**
     * @brief Creates a Pose from a x, y, z, r, t, s
     *
     * @param pose The Pose array
     * @param radians true = rotation is in Radians, false = rotation is in Degrees
     */
    ScanPose(double x, double y, double z, double r, double t, double s, bool radians = true);

    /**
     * @brief Creates a Pose from a transformation Matrix.
     *
     * @param pose The transformation Matrix
     */
    ScanPose(const Matrix4d& pose);

    /**
     * @brief Creates a Pose from a position and a set of rotation angles
     *
     * @param position The position of the Pose
     * @param angles The rotations around the x, y, z axis as x, y, z components of a Vector
     * @param radians true = angles is in Radians, false = angles is in Degrees
     */
    ScanPose(const Vector3d& position, const Vector3d& angles, bool radians = true);

	static ScanPose fromFile(const std::string& file);

	/**
	 * @brief Converts the Pose into a transformation Matrix
	 */
    Matrix4d toMatrix() const;

	/**
	 * @brief Converts the Pose into a position and angles Vector
     * 
     * @param radians true if you want the angles to be in Radians, false => Degrees
	 */
    std::pair<Vector3d, Vector3d> toPosAngle(bool radians = true) const;

	/**
	 * @brief Converts the Pose into a Pose array
     * 
     * @param radians true if you want the angles to be in Radians, false => Degrees
	 */
    std::array<double, 6> toArray(bool radians = true) const;

	/**
	 * @brief Converts the Pose into a Pose array
     * 
     * @param radians true if you want the angles to be in Radians, false => Degrees
	 */
    std::array<float, 6> toFloatArray(bool radians = true) const;

private:
    Vector3d m_position;
    Vector3d m_angles;
};

} /* namespace lvr2 */

#endif /* SCANPOSE_HPP_ */
