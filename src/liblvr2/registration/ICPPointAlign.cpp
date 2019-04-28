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
#include <lvr2/geometry/Matrix4.hpp>

// TODO: remove
#include <chrono>

#include <fstream>
using std::ofstream;

using namespace std;

namespace lvr2
{

ICPPointAlign::ICPPointAlign(PointBufferPtr model, PointBufferPtr data, const Matrix4d& modelPose, const Matrix4d& dataPose) :
    m_dataCloud(data), m_transformation(dataPose)
{
    // Init default values
    m_epsilon               = 0.00001;
    m_maxDistanceMatch      = 25;
    m_maxIterations         = 50;

    // Transform model points according to initial pose
    size_t n = model->numPoints();
    floatArr o_points = model->getPointArray();

    PointArray modelPoints = PointArray(new Vector3d[n]);

    for (size_t i = 0; i < n; i++)
    {
        Eigen::Vector4d v(o_points[3 * i], o_points[3 * i + 1], o_points[3 * i + 2], 1.0);
        modelPoints[i] = (modelPose * v).block<3, 1>(0, 0);
    }

    // Create search tree
    m_searchTree = KDTree::create(modelPoints, n);
}

Matrix4d ICPPointAlign::match()
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

        PointPairVector pairs;
        auto pre_pair = clock();
        getPointPairs(pairs, centroid_m, centroid_d);
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

        cout << timestamp << "ICP Error is " << ret << " in iteration " << i << " / " << m_maxIterations << " using " << pairs.size() << " points." << endl;

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

void ICPPointAlign::getPointPairs(PointPairVector& pairs, Vector3d& centroid_m, Vector3d& centroid_d)
{
    FloatChannel data_pts = m_dataCloud->getFloatChannel("points").get();
    size_t n = data_pts.numElements();

    pairs.clear();
    pairs.reserve(n);

    #pragma omp parallel for
    for (size_t i = 0; i < n; i++)
    {
        Eigen::Vector3d data = data_pts[i];
        Eigen::Vector4d extended(data.x(), data.y(), data.z(), 1.0);
        Eigen::Vector3d point = (m_transformation * extended).block<3, 1>(0, 0);

        Vector3d* neighbor;
        double distance;

        m_searchTree->nearestNeighbor(point, neighbor, distance, m_maxDistanceMatch);

        if (neighbor != nullptr)
        {
            #pragma omp critical
            {
                centroid_m += point;
                centroid_d += *neighbor;

                pairs.push_back(make_pair(point, *neighbor));
            }
        }
    }

    centroid_m /= pairs.size();
    centroid_d /= pairs.size();
}

ICPPointAlign::~ICPPointAlign()
{
    // TODO Auto-generated destructor stub
}

void ICPPointAlign::setMaxMatchDistance(double d)
{
    m_maxDistanceMatch = d;
}

void ICPPointAlign::setMaxIterations(int i)
{
    m_maxIterations = i;
}

void ICPPointAlign::setEpsilon(double e)
{
    m_epsilon = e;
}

double ICPPointAlign::getEpsilon()
{
    return m_epsilon;
}

double ICPPointAlign::getMaxMatchDistance()
{
    return m_maxDistanceMatch;
}


int ICPPointAlign::getMaxIterations()
{
    return m_maxIterations;
}

} /* namespace lvr2 */
