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
 * EulerPointAlign.cpp
 *
 *  @date Feb 21, 2014
 *  @author Thomas Wiemann
 */
#include <lvr2/registration/EulerPointAlign.hpp>

#include <limits>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/SVD>

using namespace Eigen;
using std::numeric_limits;

typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;
#pragma omp declare reduction (+: Vector3d: omp_out=omp_out+omp_in) initializer(omp_priv=Vector3d(0, 0, 0))

namespace lvr2
{

template <typename BaseVecT>
Matrix4d EulerPointAlign<BaseVecT>::alignPoints(const PointPairVector<BaseVecT>& pairs, const Matrix4<BaseVecT>& old_transform)
{
    float pose[6];
    Matrix4<BaseVecT> transposed(old_transform);
    transposed.transpose();
    transposed.toPostionAngle(pose);

    Vector3d pos(pose[0], pose[1], pose[2]);
    Vector3d theta(pose[3], pose[4], pose[5]);

    /// create transform matrix of the first scan
    Matrix4d trans_m = Matrix4d::Identity();
    trans_m.block<3, 3>(0, 0) = AngleAxisd(theta.x(), Vector3d::UnitX()).matrix()
                                * AngleAxisd(theta.y(), Vector3d::UnitY())
                                * AngleAxisd(theta.z(), Vector3d::UnitZ());

    trans_m.block<3, 1>(0, 3) = pos;

    Vector3d sin, cos;
    sincos(theta.x(), &sin.x(), &cos.x());
    sincos(theta.y(), &sin.y(), &cos.y());
    sincos(theta.z(), &sin.z(), &cos.z());

    /// create matrix H
    Matrix6d H = Matrix6d::Identity();
    H(0, 4) = -pos.z() * cos.x() + pos.y() * sin.x();
    H(0, 5) = pos.y() * cos.x() * cos.y() + pos.z() * cos.y() * sin.x();
    H(1, 3) = pos.z();
    H(1, 4) = -pos.x() * sin.x();
    H(1, 5) = -pos.x() * cos.x() * cos.y() + pos.z() * sin.y();
    H(2, 3) = -pos.y();
    H(2, 4) = pos.x() * cos.x();
    H(2, 5) = -pos.x() * cos.y() * sin.x() - pos.y() * sin.y();
    H(3, 5) = sin.y();
    H(4, 4) = sin.x();
    H(4, 5) = cos.x() * cos.y();
    H(5, 4) = cos.x();
    H(5, 5) = -cos.y() * sin.x();



    Vector3d mid_sum = Vector3d::Zero();
    Vector3d mz_top = Vector3d::Zero(), mz_bottom = Vector3d::Zero();
    double xpy = 0.0, xpz = 0.0, ypz = 0.0;
    double xy = 0.0, xz = 0.0, yz = 0.0;
    #pragma omp parallel for reduction(+:mid_sum, xpy, xpz, ypz, xy, xz, yz, mz_top, mz_bottom)
    for (size_t i = 0; i < pairs.size(); i++)
    {
        const BaseVecT& base_d = pairs[i].first;
        const BaseVecT& base_m = pairs[i].second;

        Vector3d pd(base_d[0], base_d[1], base_d[2]);
        Vector3d pm(base_m[0], base_m[1], base_m[2]);

        Vector3d mid = (pm + pd) / 2.0;
        Vector3d delta = pm - pd;

        mid_sum += mid;

        /// sums of squares of pairs of coordinates
        xpy += mid.x() * mid.x() + mid.y() * mid.y();
        xpz += mid.x() * mid.x() + mid.z() * mid.z();
        ypz += mid.y() * mid.y() + mid.z() * mid.z();

        /// sums of products of pairs of coordinates
        xy += mid.x() * mid.y();
        xz += mid.x() * mid.z();
        yz += mid.y() * mid.z();

        mz_top += delta;
        mz_bottom += Vector3d(-mid.z() * delta.y() + mid.y() * delta.z(),
                              -mid.y() * delta.x() + mid.x() * delta.y(),
                              mid.z() * delta.x() - mid.x() * delta.z());
    }

    Vector6d mz;
    mz.block<3, 1>(0, 0) = mz_top;
    mz.block<3, 1>(3, 0) = mz_bottom;

    Matrix6d mm = Matrix6d::Identity();
    mm(0, 0) = mm(1, 1) = mm(2, 2) = pairs.size();
    mm(3, 3) = ypz;
    mm(4, 4) = xpy;
    mm(5, 5) = xpz;
    mm(0, 4) = mm(4, 0) = -mid_sum.y();
    mm(0, 5) = mm(5, 0) = mid_sum.z();
    mm(1, 3) = mm(3, 1) = -mid_sum.z();
    mm(1, 4) = mm(4, 1) = mid_sum.x();
    mm(2, 3) = mm(3, 2) = mid_sum.y();
    mm(2, 5) = mm(5, 2) = -mid_sum.x();
    mm(3, 4) = mm(4, 3) = -xz;
    mm(3, 5) = mm(5, 3) = -xy;
    mm(4, 5) = mm(5, 4) = -yz;

    Vector6d eHat = mm.inverse() * mz;

    /// the vector pose_d is the pose estimation of the
    // second scan = the final pose of the first scan
    Vector6d pose_d;
    pose_d.block<3, 1>(0, 0) = pos;
    pose_d.block<3, 1>(3, 0) = theta;

    pose_d -= H.inverse() * eHat;

    /// transform of the second scan as computed so far
    Matrix4d trans_d = Matrix4d::Identity();
    trans_d.block<3, 3>(0, 0) = AngleAxisd(pose_d(3), Vector3d::UnitX()).matrix()
                                * AngleAxisd(pose_d(4), Vector3d::UnitY())
                                * AngleAxisd(pose_d(5), Vector3d::UnitZ());

    trans_d.block<3, 1>(0, 3) = pose_d.block<3, 1>(0, 0);

    /// the incremental transform calculated from the absolute poses
    //  of the two scans
    Matrix4d result = trans_m * trans_d.inverse();

    return result.transpose();
}

} // namespace lvr2
