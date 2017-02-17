#include <lvr/reconstruction/PanoramaNormals.hpp>
#include <lvr/geometry/Normal.hpp>
#include <lvr/io/Progress.hpp>
#include <lvr/io/Timestamp.hpp>

#include <Eigen/Dense>

#include <iostream>
#include <iterator>
#include <algorithm>

#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>

using std::cout;
using std::endl;

namespace lvr
{

PanoramaNormals::PanoramaNormals(ModelToImage* mti)
    : m_mti(mti)
{

}

PointBufferPtr PanoramaNormals::computeNormals(int width, int height, bool interpolate)
{
    // Create new point buffer and tmp storages
    PointBufferPtr out_buffer(new PointBuffer);
    vector<float> pts;
    vector<float> normals;

    // Get input buffer's points
    PointBufferPtr in_buffer = m_mti->pointBuffer();
    size_t n_inPoints;
    floatArr in_points = in_buffer->getPointArray(n_inPoints);

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
                Vertex<float> mean;
                for(int i = 0; i < nb.size(); i++)
                {
                    // Determine position of geometry in point array
                    size_t index = nb[i].index * 3;

                    // Get point coordinates
                    Vertex<float> neighbor(in_points[index],
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

                    Vertex<float> pt(in_points[index] - mean.x,
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

                Normal<float> nn(nx, ny, nz);
                Vertex<float> center(0, 0, 0);

                size_t index = mat.pixels[i][j][0].index * 3;
                Vertex<float> p1 = center - Vertex<float>(in_points[index], in_points[index + 1], in_points[index + 2]);
//                float angle = atan2(p1.cross(nn).length(), p1 * nn);

//                if(angle > M_PI/2 || angle < -M_PI/2)
//                {
//                    nx *= -1;
//                    ny *= -1;
//                    nz *= -1;
//                }

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


                    pts.push_back(in_points[index]);
                    pts.push_back(in_points[index + 1]);
                    pts.push_back(in_points[index + 2]);

                    if(!interpolate)
                    {
                        normals.push_back(nx);
                        normals.push_back(ny);
                        normals.push_back(nz);
                    }
                }
            }
        }
        ++progress;
    }
    cout << endl;

    if(interpolate)
    {
        cout << timestamp << " Interpolating normals" << endl;
        for(size_t i = 0; i < mat.pixels.size(); i++)
        {
            for(size_t j = 0; j < mat.pixels[i].size(); j++)
            {
                float x = 0.0f;
                float y = 0.0f;
                float z = 0.0f;
                int np = 0;
                for(int off_i = -di; off_i <= di; off_i++)
                {
                    for(int off_j = -dj; off_j <= dj; off_j++)
                    {
                        int p_i = i + off_i;
                        int p_j = j + off_j;


                        if(p_i >= 0 && p_i < mat.pixels.size() &&
                           p_j >= 0 && p_j < mat.pixels[i].size())
                        {
                            for(size_t k = 0; k < mat.pixels[p_i][p_j].size(); k++)
                            {
                                size_t index = mat.pixels[p_i][p_j][k].index;
                                x += in_points[index];
                                y += in_points[index + 1];
                                z += in_points[index + 2];
                                np++;
                            }
                        }
                    }
                }
                if(np > 3) // Same condition as above
                {
                    x /= np;
                    y /= np;
                    z /= np;
                    normals.push_back(x);
                    normals.push_back(y);
                    normals.push_back(z);
                }

            }
        }

        cout << normals.size() << " " << pts.size() << endl;
    }


    // Generate point buffer
    floatArr p_arr(new float[pts.size()]);
    floatArr n_arr(new float[normals.size()]);


    for(size_t i = 0; i < pts.size(); i++)
    {
        p_arr[i] = pts[i];
        n_arr[i] = normals[i];
    }

    cout << pts.size() << endl;

    out_buffer->setPointArray(p_arr, pts.size() / 3);
    out_buffer->setPointNormalArray(n_arr, normals.size() / 3);

    return out_buffer;

}

} // namespace lvr

