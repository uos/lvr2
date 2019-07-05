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
    Vector3f* data,
    Vector3f** neighbors,
    size_t n,
    const Vector3f& centroid_m,
    const Vector3f& centroid_d,
    Matrix4f& align)
{
    double error = 0.0;
    size_t pairs = 0;

    // Fill H matrix
    Matrix3f H = Matrix3f::Zero();

    #pragma omp parallel
    {
        Matrix3f localH = Matrix3f::Zero();
        double localError = 0.0;
        size_t localPairs = 0;

        #pragma omp for nowait
        for (size_t i = 0; i < n; i++)
        {
            if (neighbors[i] == nullptr)
            {
                continue;
            }

            Vector3f m = *neighbors[i] - centroid_m;
            Vector3f d = data[i] - centroid_d;

            localError += (m - d).squaredNorm();
            localPairs++;

            // same as "localH += m * d.transpose();" but faster
            for (int j = 0; j < 3; j++)
            {
                for (int k = 0; k < 3; k++)
                {
                    localH(j, k) += d[j] * m[k];
                }
            }
        }

        #pragma omp critical
        {
            H += localH;
            error += localError;
            pairs += localPairs;
        }
    }

    error = sqrt(error / (double)pairs);

    JacobiSVD<Matrix3f> svd(H, ComputeFullU | ComputeFullV);

    Matrix3f U = svd.matrixU();
    Matrix3f V = svd.matrixV();

    Matrix3f R = V * U.transpose();
    align.block<3, 3>(0, 0) = R;

    // Calculate translation
    Eigen::Vector3f translation = centroid_m - R * centroid_d;
    align.block<3, 1>(0, 3) = translation;

    return error;
}

} // namespace lvr2
