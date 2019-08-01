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

namespace lvr2
{

double EigenSVDPointAlign::alignPoints(
    ScanPtr scan,
    Vector3f** neighbors,
    const Vector3d& centroid_m,
    const Vector3d& centroid_d,
    Matrix4d& align)
{
    double error = 0.0;
    size_t pairs = 0;

    // Fill H matrix
    Matrix3d H = Matrix3d::Zero();

    for (size_t i = 0; i < scan->count(); i++)
    {
        if (neighbors[i] == nullptr)
        {
            continue;
        }

        Vector3d m = neighbors[i]->cast<double>() - centroid_m;
        Vector3d d = scan->getPoint(i).cast<double>() - centroid_d;

        error += (m - d).squaredNorm();
        pairs++;

        // same as "localH += m * d.transpose();" but faster
        for (int j = 0; j < 3; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                H(j, k) += d[j] * m[k];
            }
        }
    }

    error = sqrt(error / (double)pairs);

    JacobiSVD<Matrix3d> svd(H, ComputeFullU | ComputeFullV);

    Matrix3d U = svd.matrixU();
    Matrix3d V = svd.matrixV();

    Matrix3d R = V * U.transpose();

    align = Matrix4d::Identity();
    align.block<3, 3>(0, 0) = R;

    // Calculate translation
    Eigen::Vector3d translation = centroid_m - R * centroid_d;
    align.block<3, 1>(0, 3) = translation;

    return error;
}

double EigenSVDPointAlign::alignPoints(
    PointPairVector& pairs,
    const Vector3d& centroid_m,
    const Vector3d& centroid_d,
    Matrix4d& align)
{
    double error = 0.0;
    size_t n = pairs.size();

    // Fill H matrix
    Matrix3d H = Matrix3d::Zero();

    for (size_t i = 0; i < n; i++)
    {
        Vector3d m = pairs[i].first.cast<double>() - centroid_m;
        Vector3d d = pairs[i].second.cast<double>() - centroid_d;

        error += (m - d).squaredNorm();

        // same as "localH += m * d.transpose();" but faster
        for (int j = 0; j < 3; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                H(j, k) += d[j] * m[k];
            }
        }
    }

    error = sqrt(error / (double)n);

    JacobiSVD<Matrix3d> svd(H, ComputeFullU | ComputeFullV);

    Matrix3d U = svd.matrixU();
    Matrix3d V = svd.matrixV();

    Matrix3d R = V * U.transpose();

    align = Matrix4d::Identity();
    align.block<3, 3>(0, 0) = R;

    // Calculate translation
    Eigen::Vector3d translation = centroid_m - R * centroid_d;
    align.block<3, 1>(0, 3) = translation;

    return error;
}

} // namespace lvr2
