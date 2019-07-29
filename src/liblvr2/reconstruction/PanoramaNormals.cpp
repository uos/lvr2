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

#include "lvr2/reconstruction/PanoramaNormals.hpp"

#include "lvr2/geometry/BaseVector.hpp"
#include "lvr2/geometry/Normal.hpp"
#include "lvr2/io/Progress.hpp"
#include "lvr2/io/Timestamp.hpp"

#include <Eigen/Dense>

#include <iostream>
#include <iterator>
#include <algorithm>

#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>

using std::cout;
using std::endl;

namespace lvr2
{

using Vec = BaseVector<float>;

PanoramaNormals::PanoramaNormals(ModelToImage* mti)
    : m_mti(mti)
{
    m_buffer = mti->pointBuffer();
}

PointBufferPtr PanoramaNormals::computeNormals(int width, int height, bool interpolate)
{
    // Create new point buffer and tmp storages
    PointBufferPtr out_buffer(new PointBuffer);
    vector<float> pts;
    vector<float> normals;

    // Get input buffer's points
    PointBufferPtr in_buffer = m_mti->pointBuffer();
    size_t w_color;
    size_t n_inPoints = in_buffer->numPoints();
    floatArr in_points = in_buffer->getPointArray();
    ucharArr in_colors = in_buffer->getColorArray(w_color);

    // Reserve memory for output buffers (we need a deep copy)
    floatArr p_arr(new float[n_inPoints * 3]);
    floatArr n_arr(new float[n_inPoints * 3]);
    ucharArr c_arr;
    if(in_buffer->hasColors())
    {
        c_arr = ucharArr(new unsigned char[n_inPoints * 3]);
    }

    // Get panorama
    ModelToImage::DepthListMatrix mat;
    m_mti->computeDepthListMatrix(mat);

    // If the desired neighborhood is larger than 2 x 2 pixels
    // compute offsets for i und j dimension of the image.
    int di = 2;
    if(width > 2)
    {
        di = width / 2;
    }


    int dj = 2;
    if(height > 2)
    {
        dj = height / 2;
    }

    // Compute normals
    // Create progress output
    string comment = timestamp.getElapsedTime() + "Computing normals ";
    ProgressBar progress(mat.pixels.size(), comment);



    for(size_t i = 0; i < mat.pixels.size(); i++)
    {
        #pragma omp parallel for
        for(size_t j = 0; j < mat.pixels[i].size(); j++)
        {
            // Check if image entry is empty
            if(mat.pixels[i][j].size() == 0)
            {
                continue;
            }

            // Collect 'neighboring' points
            vector<ModelToImage::PanoramaPoint> nb;

            // The points at the current position are part of the neighborhood
            std::copy(mat.pixels[i][j].begin(), mat.pixels[i][j].end(), std::back_inserter(nb));

            for(int off_i = -di; off_i <= di; off_i++)
            {
                for(int off_j = -dj; off_j <= dj; off_j++)
                {
                    int p_i = i + off_i;
                    int p_j = j + off_j;


                    if(p_i >= 0 && p_i < mat.pixels.size() &&
                       p_j >= 0 && p_j < mat.pixels[i].size())
                    {
                        // We only save the first point as representative
                        // because using all points from list will likely
                        // result in undesirable configurations for local
                        // normal estimation
                        if(mat.pixels[p_i][p_j].size() > 0)
                        {
                            nb.push_back(mat.pixels[p_i][p_j][0]);
                        }
                    }
                }
            }

            // Compute normal if more than three neighbors where found
            if(nb.size() > 3)
            {


                // Compute mean
                Vec mean;
                for(int i = 0; i < nb.size(); i++)
                {
                    // Determine position of geometry in point array
                    size_t index = nb[i].index * 3;

                    // Get point coordinates
                    Vec neighbor(in_points[index],
                                           in_points[index + 1],
                                           in_points[index + 2]);

                    // Add to mean
                    mean.x += neighbor.x;
                    mean.y += neighbor.y;
                    mean.z += neighbor.z;
                }
                mean.x /= nb.size();
                mean.y /= nb.size();
                mean.z /= nb.size();

                // Calculate covariance
                double covariance[9] = {0};

                for(int i = 0; i < nb.size(); i++)
                {
                    size_t index = nb[i].index * 3;

                    Vec pt(in_points[index    ] - mean.x,
                                     in_points[index + 1] - mean.y,
                                     in_points[index + 3] - mean.z);

                    covariance[4] += pt.y * pt.y;
                    covariance[7] += pt.y * pt.z;
                    covariance[8] += pt.z * pt.z;

                    pt.x *= pt.x;
                    pt.y *= pt.x;
                    pt.z *= pt.x;

                    covariance[0] += pt.x;
                    covariance[1] += pt.y;
                    covariance[6] += pt.z;

                }

                covariance[3] = covariance[1];
                covariance[2] = covariance[6];
                covariance[5] = covariance[7];

                for(int i = 0; i < 9; i++)
                {
                    covariance[i] /= nb.size();
                }

                // Compute eigenvalues and eigenvectors using GSL
                gsl_matrix_view m = gsl_matrix_view_array(covariance, 3, 3);
                gsl_matrix* evec = gsl_matrix_alloc(3, 3);
                gsl_vector* eval = gsl_vector_alloc(3);


                gsl_eigen_symmv_workspace * w = gsl_eigen_symmv_alloc (3);
                gsl_eigen_symmv (&m.matrix, eval, evec, w);

                gsl_eigen_symmv_free (w);
                gsl_eigen_symmv_sort (eval, evec, GSL_EIGEN_SORT_ABS_ASC);

                gsl_vector_view evec_0 = gsl_matrix_column(evec, 0);
                float nx = gsl_vector_get(&evec_0.vector, 0);
                float ny = gsl_vector_get(&evec_0.vector, 1);
                float nz = gsl_vector_get(&evec_0.vector, 2);

                // Flip normals towards reference point
                Normal<float> nn(nx, ny, nz);
                Vec center(0, 0, 0);

                size_t index = mat.pixels[i][j][0].index * 3;
                Vec p1 = center - Vec(in_points[index], in_points[index + 1], in_points[index + 2]);

                if(Normal<float>(p1) * nn < 0)
                {
                    nx *= -1;
                    ny *= -1;
                    nz *= -1;
                }

                for(size_t k = 0; k < mat.pixels[i][j].size(); k++)
                {
                    // Assign the same normal to all points
                    // behind this pixel to preserve the complete
                    // point cloud
                    size_t index = mat.pixels[i][j][k].index * 3;
                    size_t color_index = mat.pixels[i][j][k].index * w_color;

                    // Copy point and normal to target buffer
                    p_arr[index    ] = in_points[index];
                    p_arr[index + 1] = in_points[index + 1];
                    p_arr[index + 2] = in_points[index + 2];

                    if(in_buffer->hasColors())
                    {
                        c_arr[index    ] = in_colors[color_index];
                        c_arr[index + 1] = in_colors[color_index + 1];
                        c_arr[index + 2] = in_colors[color_index + 2];
                    }

                    if(!interpolate)
                    {
                        n_arr[index    ] = nx;
                        n_arr[index + 1] = ny;
                        n_arr[index + 2] = nz;
                    }
                }
            }
        }
        ++progress;
    }
    cout << endl;

//    if(interpolate)
//    {
//        cout << timestamp << " Interpolating normals" << endl;
//        for(size_t i = 0; i < mat.pixels.size(); i++)
//        {
//            for(size_t j = 0; j < mat.pixels[i].size(); j++)
//            {
//                float x = 0.0f;
//                float y = 0.0f;
//                float z = 0.0f;
//                int np = 0;
//                for(int off_i = -di; off_i <= di; off_i++)
//                {
//                    for(int off_j = -dj; off_j <= dj; off_j++)
//                    {
//                        int p_i = i + off_i;
//                        int p_j = j + off_j;


//                        if(p_i >= 0 && p_i < mat.pixels.size() &&
//                           p_j >= 0 && p_j < mat.pixels[i].size())
//                        {
//                            for(size_t k = 0; k < mat.pixels[p_i][p_j].size(); k++)
//                            {
//                                size_t index = mat.pixels[p_i][p_j][k].index;
//                                x += in_points[index];
//                                y += in_points[index + 1];
//                                z += in_points[index + 2];
//                                np++;
//                            }
//                        }
//                    }
//                }
//                if(np > 3) // Same condition as above
//                {
//                    x /= np;
//                    y /= np;
//                    z /= np;
//                    normals.push_back(x);
//                    normals.push_back(y);
//                    normals.push_back(z);
//                }

//            }
//        }

//        cout << normals.size() << " " << pts.size() << endl;
//    }

    cout << timestamp << "Finished normal estimation" << endl;

    if(in_buffer->hasColors())
    {
        out_buffer->setColorArray(c_arr, n_inPoints);
    }
    out_buffer->setPointArray(p_arr, n_inPoints);
    out_buffer->setNormalArray(n_arr, n_inPoints);

    return out_buffer;
}

} // namespace lvr2
