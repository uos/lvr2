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
 * EigenSVDPointAlign.cpp
 *
 *  @date Feb 21, 2014
 *  @author Thomas Wiemann
 */
#include <lvr2/registration/EigenSVDPointAlign.hpp>
#include <lvr2/io/IOUtils.hpp>

#include <limits>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/SVD>

using namespace Eigen;
using std::numeric_limits;

using Matrix6f = Matrix<float, 6, 6>;
using Vector6f = Matrix<float, 6, 1>;

namespace lvr2
{

double EigenSVDPointAlign::alignPoints(
    const PointPairVector& pairs,
    const Vector3f& centroid_m,
    const Vector3f& centroid_d,
    Matrix4f& align)
{
/*
    Vector3f pos, theta;
    matrixToPose(align, pos, theta);
    
    /// create transform matrix of the first scan
    Matrix4f trans_m = poseToMatrix(pos, theta);

    Vector3f sin, cos;
    sincos(theta.x(), &sin.x(), &cos.x());
    sincos(theta.y(), &sin.y(), &cos.y());
    sincos(theta.z(), &sin.z(), &cos.z());

    /// create matrix H
    Matrix6f H = Matrix6f::Identity();
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

    #pragma omp declare reduction (+: Vector3f: omp_out=omp_out+omp_in) initializer(omp_priv=Vector3f::Zero())
    #pragma omp declare reduction (+: Vector6f: omp_out=omp_out+omp_in) initializer(omp_priv=Vector6f::Zero())

    Vector3f mid_sum = Vector3f::Zero();
    Vector6f mz = Vector6f::Zero();
    double xpy = 0.0, xpz = 0.0, ypz = 0.0;
    double xy = 0.0, xz = 0.0, yz = 0.0;
    double error = 0.0;
    #pragma omp parallel for reduction(+:mid_sum, xpy, xpz, ypz, xy, xz, yz, mz, error)
    for (size_t i = 0; i < pairs.size(); i++)
    {
        Vector3f mid = (*pairs[i].second + *pairs[i].first) / 2.0;
        Vector3f delta = *pairs[i].second - *pairs[i].first;

        mid_sum += mid;

        /// sums of squares of pairs of coordinates
        xpy += mid.x() * mid.x() + mid.y() * mid.y();
        xpz += mid.x() * mid.x() + mid.z() * mid.z();
        ypz += mid.y() * mid.y() + mid.z() * mid.z();

        /// sums of products of pairs of coordinates
        xy += mid.x() * mid.y();
        xz += mid.x() * mid.z();
        yz += mid.y() * mid.z();

        mz.block<3, 1>(0, 0) += delta;
        mz.block<3, 1>(3, 0) += Vector3f(-mid.z() * delta.y() + mid.y() * delta.z(),
                              -mid.y() * delta.x() + mid.x() * delta.y(),
                              mid.z() * delta.x() - mid.x() * delta.z());
        
        error += delta.squaredNorm();
    }

    error = sqrt(error / pairs.size());

    Matrix6f mm = Matrix6f::Identity();
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

    Vector6f eHat = mm.inverse() * mz;

    /// the vector pose_d is the pose estimation of the
    // second scan = the final pose of the first scan
    Vector6f pose_d;
    pose_d.block<3, 1>(0, 0) = pos;
    pose_d.block<3, 1>(3, 0) = theta;

    pose_d -= H.inverse() * eHat;

    /// transform of the second scan as computed so far
    Matrix4f trans_d = poseToMatrix(pose_d.block<3, 1>(0, 0), pose_d.block<3, 1>(3, 0));

    /// the incremental transform calculated from the absolute poses
    //  of the two scans
    align = trans_m * trans_d.inverse();

    return error;
// */
//*
    double error = 0.0;

    // Fill H matrix
    Matrix3f H = Matrix3f::Zero();

    // openMP seems to make this part slower
    // #pragma omp declare reduction (+: Matrix3f: omp_out=omp_out+omp_in) initializer(omp_priv=Matrix3f::Zero())
    // #pragma omp parallel for reduction(+:H)
    for (size_t i = 0; i < pairs.size(); i++)
    {
        Vector3f m = *pairs[i].first - centroid_m;
        Vector3f d = *pairs[i].second - centroid_d;

        error += (m - d).squaredNorm();

        // same as "H += m[i] * d[i].transpose();" but faster
        for (int j = 0; j < 3; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                H(j, k) += m[j] * d[k];
            }
        }
    }

    error = sqrt(error / (double)pairs.size());

    JacobiSVD<Matrix3f> svd(H, ComputeFullU | ComputeFullV);

    Matrix3f U = svd.matrixU();
    Matrix3f V = svd.matrixV();

    Matrix3f R = V * U.transpose();
    align.block<3, 3>(0, 0) = R;

    // Calculate translation
    Eigen::Vector3f translation = centroid_d - R * centroid_m;
    align.block<3, 1>(0, 3) = translation;

    return error;
// */
}

} // namespace lvr2
