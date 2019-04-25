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
 * ICPPointAlign.cpp
 *
 *  @date Mar 18, 2014
 *  @author Thomas Wiemann
 */
#include <lvr2/registration/ICPPointAlign.hpp>
#include <lvr2/registration/EigenSVDPointAlign.hpp>
#include <lvr2/io/Timestamp.hpp>
#include <lvr2/io/IOUtils.hpp>

// TODO: remove
#include <chrono>

#include <fstream>
using std::ofstream;

namespace lvr2
{

template <typename BaseVecT>
ICPPointAlign<BaseVecT>::ICPPointAlign(PointBufferPtr model, PointBufferPtr data, const Matrix4d& modelPose, const Matrix4d& dataPose) :
    m_dataCloud(data), m_transformation(dataPose)
{
    // Init default values
    m_epsilon               = 0.00001;
    m_maxDistanceMatch      = 25;
    m_maxIterations         = 50;

    // Transform model points according to initial pose
    m_modelCloud = PointBufferPtr(new PointBuffer);
    size_t n = model->numPoints();
    floatArr o_points = model->getPointArray();
    floatArr t_points(new float[3 * n]);

    for (size_t i = 0; i < n; i++)
    {
        Eigen::Vector4d v(o_points[3 * i], o_points[3 * i + 1], o_points[3 * i + 2], 1.0);
        Eigen::Vector4d t = modelPose * v;
        t_points[3 * i    ] = t.x();
        t_points[3 * i + 1] = t.y();
        t_points[3 * i + 2] = t.z();
    }
    m_modelCloud->setPointArray(t_points, n);

    // Create search tree
    m_searchTree = make_shared<SearchTreeFlann<BaseVecT>>(m_modelCloud);

}

template <typename BaseVecT>
Matrix4d ICPPointAlign<BaseVecT>::match()
{
    if (m_maxIterations == 0)
    {
        return Matrix4d();
    }

    auto start_time = clock();
    double pairTime = 0;
    double alignTime = 0;

    double ret = 0.0, prev_ret = 0.0, prev_prev_ret = 0.0;
    EigenSVDPointAlign align;
    for (int i = 0; i < m_maxIterations; i++)
    {
        // Update break variables
        prev_prev_ret = prev_ret;
        prev_ret = ret;

        // Get point pairs
        Vector3d centroid_m = Vector3d::Zero();
        Vector3d centroid_d = Vector3d::Zero();
        Matrix4d transform;
        double sum;


        PointPairVector pairs;
        auto pre_pair = clock();
        getPointPairs(pairs, centroid_m, centroid_d, sum);
        pairTime += (double)(clock() - pre_pair) / CLOCKS_PER_SEC / 5.0;

        // Get transformation (if possible)
        auto pre_align = clock();
        ret = align.alignPoints(pairs, centroid_m, centroid_d, transform);
        alignTime += (double)(clock() - pre_align) / CLOCKS_PER_SEC / 5.0;

        //cout << timestamp << "CORRECTION" << endl;
        //cout << transform << endl;

        // Apply transformation
        m_transformation = transformRegistration(m_transformation, transform);

        //cout << timestamp << "TRANSFORMATION: " << endl;
        //cout << m_transformation << endl;

        cout << timestamp << "ICP Error is " << ret << " in iteration " << i << " / " << m_maxIterations << " using " << pairs.size() << " points."<< endl;

        // Check minimum distance
        if ((fabs(ret - prev_ret) < m_epsilon) && (fabs(ret - prev_prev_ret) < m_epsilon))
        {
            cout << timestamp << " Error below m_epsilon after " << i << " / " << m_maxIterations << " Iterations" << endl;
            break;
        }
    }
    cout << "Time: " << (double)(clock() - start_time) / CLOCKS_PER_SEC / 5.0 << endl;
    cout << "Pairing time: " << pairTime << "; " << "Algining time: " << alignTime << endl;
    cout << "Error: " << ret << "; Result: " << endl << m_transformation << endl;
    return m_transformation;
}

template <typename BaseVecT>
void ICPPointAlign<BaseVecT>::getPointPairs(PointPairVector& pairs, Vector3d& centroid_m, Vector3d& centroid_d, double& sum)
{
    sum = 0;

    FloatChannel model_pts = m_modelCloud->getFloatChannel("points").get();
    FloatChannel data_pts = m_dataCloud->getFloatChannel("points").get();
    size_t n = data_pts.numElements();

    BaseVecT* transformed = new BaseVecT[n];
    Matrix4<BaseVecT> transform;
    transform = m_transformation.transpose();
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++)
    {
        BaseVecT data = data_pts[i];
        transformed[i] = transform * data;
    }

    size_t* indices = new size_t[n];
    float* distances = new float[n];
    m_searchTree->kSearchMany(transformed, n, 1, indices, distances);

    pairs.clear();
    pairs.reserve(n);
    for (size_t i = 0; i < n; i++)
    {
        if (distances[i] < m_maxDistanceMatch * m_maxDistanceMatch)
        {
            Vector3d t(transformed[i].x, transformed[i].y, transformed[i].z);
            Vector3d closest = model_pts[indices[i]];

            centroid_m += t;
            centroid_d += closest;

            sum += distances[i];

            pairs.push_back(make_pair(t, closest));
        }
    }

    centroid_m /= pairs.size();
    centroid_d /= pairs.size();

    delete[] indices;
    delete[] distances;
}

template <typename BaseVecT>
ICPPointAlign<BaseVecT>::~ICPPointAlign()
{
    // TODO Auto-generated destructor stub
}

template <typename BaseVecT>
void ICPPointAlign<BaseVecT>::setMaxMatchDistance(double d)
{
    m_maxDistanceMatch = d;
}

template <typename BaseVecT>
void ICPPointAlign<BaseVecT>::setMaxIterations(int i)
{
    m_maxIterations = i;
}

template <typename BaseVecT>
void ICPPointAlign<BaseVecT>::setEpsilon(double e)
{
    m_epsilon = e;
}

template <typename BaseVecT>
double ICPPointAlign<BaseVecT>::getEpsilon()
{
    return m_epsilon;
}

template <typename BaseVecT>
double ICPPointAlign<BaseVecT>::getMaxMatchDistance()
{
    return m_maxDistanceMatch;
}


template <typename BaseVecT>
int ICPPointAlign<BaseVecT>::getMaxIterations()
{
    return m_maxIterations;
}

} /* namespace lvr2 */
