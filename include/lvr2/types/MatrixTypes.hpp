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
 * @file MatrixTypes.hpp
 * @author Thomas Wiemann (twiemann@uos.de)
 * @date 2019-08-14
 * 
 * @copyright Copyright (c) 2019
 * 
 */

#ifndef LVR2_TYPES_MATRIXTYPES_HPP
#define LVR2_TYPES_MATRIXTYPES_HPP

#include <Eigen/Dense>

namespace lvr2
{
/// General alias for row major 4x4 matrices 
template<typename T> 
using Matrix4RM = Eigen::Matrix<T, 4, 4, Eigen::RowMajor>;

/// 4x4 row major matrix with float scalars
using Matrix4fRM = Matrix4RM<float>;

/// 4x4 row major matrix with double scalars
using Matrix4dRM = Matrix4RM<double>;

/// General alias for row major 3x3 matrices 
template<typename T> 
using Matrix3RM = Eigen::Matrix<T, 3, 3, Eigen::RowMajor>;

/// 3x3 row major matrix with float scalars
using Matrix3fRM = Matrix3RM<float>;
/// 3x3 row major matrix with double scalars
using Matrix3dRM = Matrix3RM<double>;

/// General 4x4 transformation matrix (4x4)
template<typename T>
using Transform = Eigen::Matrix<T, 4, 4>;

/// 4x4 single precision transformation matrix
using Transformf = Transform<float>;

/// 4x4 double precision transformation matrix
using Transformd = Transform<double>;

/// General 3x3 rotation matrix
template<typename T>
using Rotation = Eigen::Matrix<T, 3, 3>;

/// Single precision 3x3 rotation matrix
using Rotationf = Rotation<float>;

/// Double precision 3x3 rotation matrix
using Rotationd = Rotation<double>;

/// 4x4 extrinsic calibration
template<typename T>
using Extrinsics = Eigen::Matrix<T, 4, 4>;

/// 4x4 extrinsic calibration (single precision)
using Extrinsicsf = Extrinsics<float>;

/// 4x4 extrinsic calibration (double precision)
using Extrinsicsd = Extrinsics<double>;

/// 4x4 extrinsic calibration
template<typename T>
using Intrinsics = Eigen::Matrix<T, 3, 3>;

/// 4x4 intrinsic calibration (single precision)
using Intrinsicsf = Intrinsics<float>;

/// 4x4 extrinsic calibration (double precision)
using Intrinsicsd = Intrinsics<double>;

/// Distortion Parameters
template<typename T>
using Distortion = Eigen::Matrix<T, 6, 1>;

/// Distortion Parameters (double precision)
using Distortiond = Distortion<double>;

/// Distortion Parameters (single precision)
using Distortionf = Distortion<float>;

/// Eigen 3D vector
template<typename T>
using Vector3 = Eigen::Matrix<T, 3, 1>;

/// Eigen 3D vector, single precision
using Vector3f = Eigen::Vector3f;

/// Eigen 3D vector, double precision
using Vector3d = Eigen::Vector3d;

/// Eigen 3D vector, integer
using Vector3i = Eigen::Vector3i;

/// Eigen 4D vector
template<typename T>
using Vector4 = Eigen::Matrix<T, 4, 1>;

/// Eigen 4D vector, single precision
using Vector4f = Eigen::Vector4f;

/// Eigen 4D vector, double precision
using Vector4d = Eigen::Vector4d;

/// Eigen 2D vector
template<typename T>
using Vector2 = Eigen::Matrix<T, 2, 1>;

/// Eigen 2D vector, single precision
using Vector2f = Eigen::Vector2f;

/// Eigen 2D vector, double precision
using Vector2d = Eigen::Vector2d;

/// Eigen 4x4 matrix, single precision
using Matrix4f = Eigen::Matrix4f;

/// Eigen 4x4 matrix, double precision
using Matrix4d = Eigen::Matrix4d;

/// 6D Matrix, single precision
using Matrix6f = Eigen::Matrix<float, 6, 6>;

/// 6D vector, single precision
using Vector6f = Eigen::Matrix<float, 6, 1>;

/// 6D matrix double precision
using Matrix6d = Eigen::Matrix<double, 6, 6>;

/// 6D vector double precision
using Vector6d = Eigen::Matrix<double, 6, 1>;

template<typename T> 
Vector3<T> multiply(const Transform<T>& transform, const Vector3<T>& p)
{
    Vector4<T> ret(p.coeff(0), p.coeff(1), p.coeff(2), 1.0);
    ret = transform * ret;
    return Vector3<T>(ret.coeff(0), ret.coeff(1), ret.coeff(2));
}

} // namespace lvr2

// additional operators




template<typename T>
lvr2::Vector3<T> operator*(const lvr2::Transform<T>& transform, const lvr2::Vector3<T>& p)
{
    return lvr2::multiply(transform, p);
}

#endif // LVR2_TYPES_MATRIXTYPES_HPP