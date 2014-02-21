/* Copyright (C) 2011 Uni Osnabrück
 * This file is part of the LAS VEGAS Reconstruction Toolkit,
 *
 * LAS VEGAS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * LAS VEGAS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA
 */

/**
 * EigenSVDPointAlign.cpp
 *
 *  @date Feb 21, 2014
 *  @author Thomas Wiemann
 */
#include "registration/EigenSVDPointAlign.hpp"

#include <limits>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/SVD>

using namespace Eigen;
using std::numeric_limits;

namespace lvr
{


double EigenSVDPointAlign::alignPoints(const pointPairVector& pairs,
        const Vertexf centroid_m, const Vertexf centroid_d, Matrix4f& alignfx)
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
        delete [] m[i];
        delete [] d[i];
    }
    delete [] m;
    delete [] d;

    return error;
}

}
