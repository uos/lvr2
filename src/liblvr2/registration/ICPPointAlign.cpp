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
using namespace std::chrono;

#include <omp.h>
#include <fstream>
using std::ofstream;

using namespace std;

namespace lvr2
{

ICPPointAlign::ICPPointAlign(ScanPtr model, ScanPtr data) :
    m_dataCloud(data), m_modelCloud(model), m_transformation(data->getPose()), m_deltaTransform(Matrix4d::Identity())
{
    // Init default values
    m_maxDistanceMatch  = 25;
    m_maxIterations     = 50;
    m_epsilon           = 0.00001;
    m_quiet             = false;

    // Transform model points according to initial pose
    size_t n = model->count();
    const Matrix4d& modelPose = model->getPose();

    PointArray modelPoints = PointArray(new Vector3d[n]);

    #pragma omp parallel for
    for (size_t i = 0; i < n; i++)
    {
        const Vector3d& point = model->getPoint(i);
        Eigen::Vector4d v(point.x(), point.y(), point.z(), 1.0);
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

    auto start_time = steady_clock::now();
    double pairTime = 0;
    double alignTime = 0;

    double ret = 0.0, prev_ret = 0.0, prev_prev_ret = 0.0;
    EigenSVDPointAlign align;
    int iteration;
    for (iteration = 0; iteration < m_maxIterations; iteration++)
    {
        // Update break variables
        prev_prev_ret = prev_ret;
        prev_ret = ret;

        // Get point pairs
        Vector3d centroid_m = Vector3d::Zero();
        Vector3d centroid_d = Vector3d::Zero();
        Matrix4d transform;

        PointPairVector pairs;
        getPointPairs(pairs, centroid_m, centroid_d);

        // Get transformation
        ret = align.alignPoints(pairs, centroid_m, centroid_d, transform);

        // Apply transformation
        m_transformation = transformRegistration(transform, m_transformation);
        m_deltaTransform = transformRegistration(transform, m_deltaTransform);

        if (!m_quiet)
        {
            cout << timestamp << "ICP Error is " << ret << " in iteration " << iteration << " / " << m_maxIterations << " using " << pairs.size() << " points." << endl;
        }

        // Check minimum distance
        if ((fabs(ret - prev_ret) < m_epsilon) && (fabs(ret - prev_prev_ret) < m_epsilon))
        {
            break;
        }
    }
    auto duration = steady_clock::now() - start_time;
    cout << setw(6) << (int)(duration.count() / 1e6) << " ms, ";
    cout << "Error: " << fixed << setprecision(3) << setw(7) << ret;
    if (iteration < m_maxIterations)
    {
        cout << " after " << iteration << " Iterations";
    }
    cout << endl;
    if (!m_quiet)
    {
        cout << "Result: " << endl << m_transformation << endl;
    }
    return m_transformation;
}

const Matrix4d& ICPPointAlign::getDeltaTransform() const
{
    return m_deltaTransform;
}

void ICPPointAlign::getPointPairs(PointPairVector& pairs, Vector3d& centroid_m, Vector3d& centroid_d) const
{
    size_t n = m_dataCloud->count();

    centroid_m = Vector3d::Zero();
    centroid_d = Vector3d::Zero();
    pairs.clear();
    pairs.reserve(n);

    size_t numThreads = omp_get_max_threads();
    size_t pairsPerThread = ceil((double)n / numThreads);
    PointPairVector privPairs[numThreads];
    Vector3d privCentroidM[numThreads];
    Vector3d privCentroidD[numThreads];
    for (size_t i = 0; i < numThreads; i++)
    {
        privPairs[i].reserve(pairsPerThread);
        privCentroidM[i] = Vector3d::Zero();
        privCentroidD[i] = Vector3d::Zero();
    }

    #pragma omp parallel num_threads(numThreads)
    {
        size_t thread = omp_get_thread_num();

        PointPairVector& myPairs = privPairs[thread];
        Vector3d& myCentroidM = privCentroidM[thread];
        Vector3d& myCentroidD = privCentroidD[thread];

        size_t start = thread * pairsPerThread;
        size_t end = min((thread + 1) * pairsPerThread, n);

        for (size_t i = start; i < end; i++)
        {
            Eigen::Vector3d data = m_dataCloud->getPoint(i);
            Eigen::Vector4d extended(data.x(), data.y(), data.z(), 1.0);
            Eigen::Vector3d point = (m_transformation * extended).block<3, 1>(0, 0);

            Vector3d* neighbor;
            double distance;

            m_searchTree->nearestNeighbor(point, neighbor, distance, m_maxDistanceMatch);

            if (neighbor != nullptr)
            {
                myCentroidM += point;
                myCentroidD += *neighbor;
                myPairs.push_back(make_pair(point, *neighbor));
            }
        }
    }

    for (size_t i = 0; i < numThreads; i++)
    {
        pairs.insert(pairs.end(), privPairs[i].begin(), privPairs[i].end());
        centroid_m += privCentroidM[i];
        centroid_d += privCentroidD[i];
    }

    centroid_m /= pairs.size();
    centroid_d /= pairs.size();
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
void ICPPointAlign::setQuiet(bool quiet)
{
    m_quiet = quiet;
}

double ICPPointAlign::getMaxMatchDistance() const
{
    return m_maxDistanceMatch;
}

int ICPPointAlign::getMaxIterations() const
{
    return m_maxIterations;
}

double ICPPointAlign::getEpsilon() const
{
    return m_epsilon;
}

bool ICPPointAlign::getQuiet() const
{
    return m_quiet;
}

} /* namespace lvr2 */
