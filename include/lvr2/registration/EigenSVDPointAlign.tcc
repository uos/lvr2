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

#include <Eigen/SVD>

using namespace Eigen;

namespace lvr2
{

template<typename T, typename PointT>
T EigenSVDPointAlign<T, PointT>::alignPoints(
    SLAMScanPtr scan,
    Point3** neighbors,
    const Vec3& centroid_m,
    const Vec3& centroid_d,
    Mat4& align) const
{
    T error = 0.0;
    size_t pairs = 0;

    // Fill H matrix
    Mat3 H = Matrix3d::Zero();

    for (size_t i = 0; i < scan->numPoints(); i++)
    {
        if (neighbors[i] == nullptr)
        {
            continue;
        }

        Vec3 m = neighbors[i]->template cast<T>() - centroid_m;
        Vec3 d = scan->point(i).template cast<T>() - centroid_d;

        error += (m - d).squaredNorm();
        pairs++;

        // same as "H += m * d.transpose();" but faster
        for (int j = 0; j < 3; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                H(j, k) += d[j] * m[k];
            }
        }
    }

    error = sqrt(error / (T)pairs);

    JacobiSVD<Mat3> svd(H, ComputeFullU | ComputeFullV);

    Mat3 U = svd.matrixU();
    Mat3 V = svd.matrixV();

    Mat3 R = V * U.transpose();

    align = Mat4::Identity();
    align.template block<3, 3>(0, 0) = R;

    // Calculate translation
    Vec3 translation = centroid_m - R * centroid_d;
    align.template block<3, 1>(0, 3) = translation;

    return error;
}

template<typename T, typename PointT>
T EigenSVDPointAlign<T, PointT>::alignPoints(
    PointPairVector& pairs,
    const Vec3& centroid_m,
    const Vec3& centroid_d,
    Mat4& align) const
{
    T error = 0.0;
    size_t n = pairs.size();

    // Fill H matrix
    Mat3 H = Mat3::Zero();

    for (size_t i = 0; i < n; i++)
    {
        Vec3 m = pairs[i].first.template cast<T>() - centroid_m;
        Vec3 d = pairs[i].second.template cast<T>() - centroid_d;

        error += (m - d).squaredNorm();

        // same as "H += m * d.transpose();" but faster
        for (int j = 0; j < 3; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                H(j, k) += d[j] * m[k];
            }
        }
    }

    error = sqrt(error / (T)n);

    JacobiSVD<Mat3> svd(H, ComputeFullU | ComputeFullV);

    Mat3 U = svd.matrixU();
    Mat3 V = svd.matrixV();

    Mat3 R = V * U.transpose();

    align = Mat4::Identity();
    align.template block<3, 3>(0, 0) = R;

    // Calculate translation
    Vec3 translation = centroid_m - R * centroid_d;
    align.template block<3, 1>(0, 3) = translation;

    return error;
}

} // namespace lvr2
