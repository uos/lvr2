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

#include <limits>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/SVD>

using namespace Eigen;
using std::numeric_limits;

namespace lvr2
{

template <typename BaseVecT>
double EigenSVDPointAlign<BaseVecT>::alignPoints(const PointPairVector<BaseVecT>& pairs,
        const BaseVecT centroid_m, const BaseVecT centroid_d, Matrix4<BaseVecT>& alignfx)
{
    double error = 0;
    double sum = 0.0;

    // Get centered PtPairs
    double** m = new double*[pairs.size()];
    double** d = new double*[pairs.size()];

    for(unsigned int i = 0; i <  pairs.size(); i++){
        m[i] = new double[3];
        d[i] = new double[3];
        m[i][0] = pairs[i].first.x - centroid_m[0];
        m[i][1] = pairs[i].first.y - centroid_m[1];
        m[i][2] = pairs[i].first.z - centroid_m[2];
        d[i][0] = pairs[i].second.x - centroid_d[0];
        d[i][1] = pairs[i].second.y - centroid_d[1];
        d[i][2] = pairs[i].second.z - centroid_d[2];

        sum += pow(pairs[i].first.x - pairs[i].second.x, 2)
             + pow(pairs[i].first.y - pairs[i].second.y, 2)
             + pow(pairs[i].first.z - pairs[i].second.z, 2) ;

    }

    error = sqrt(sum / (double)pairs.size());

    // Fill H matrix
    Matrix3d H, R;
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            H(i,j) = 0.0;
            R(i,j) = 0.0;
        }
    }

    for(size_t i = 0; i < pairs.size(); i++){
        for(int j = 0; j < 3; j++){
            for(int k = 0; k < 3; k++){
                H(j, k) += d[i][j]*m[i][k];
            }
        }
    }

    JacobiSVD<Matrix3d> svd(H, ComputeFullU | ComputeFullV);

    Matrix3d U = svd.matrixU();
    Matrix3d V = svd.matrixV();

    R = V * U.transpose();


    // Calculate translation
    double translation[3];


    MatrixXd col_vec(3,1);
    for(int j = 0; j < 3; j++)
        col_vec(j,0) = centroid_d[j];

    MatrixXd r_time_colVec(3,1);

    r_time_colVec = R * col_vec;
    translation[0] = centroid_m[0] - r_time_colVec(0);
    translation[1] = centroid_m[1] - r_time_colVec(1);
    translation[2] = centroid_m[2] - r_time_colVec(2);


    // Fill result
    alignfx[0] = R(0,0);
    alignfx[1] = R(1,0);
    alignfx[2] = 0;
    alignfx[2] = R(2,0);
    alignfx[3] = 0;
    alignfx[4] = R(0,1);
    alignfx[5] = R(1,1);
    alignfx[6] = R(2,1);
    alignfx[7] = 0;
    alignfx[8] = R(0,2);
    alignfx[9] = R(1,2);
    alignfx[10] = R(2,2);
    alignfx[11] = 0;
    alignfx[12] = translation[0];
    alignfx[13] = translation[1];
    alignfx[14] = translation[2];
    alignfx[15] = 1;


    for(unsigned int i = 0; i <  pairs.size(); i++){
        delete m[i];
        delete d[i];
    }
    delete[] m;
    delete[] d;

    return error;
}

} // namespace lvr2
