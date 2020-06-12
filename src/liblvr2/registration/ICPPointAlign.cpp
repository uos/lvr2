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
#include "lvr2/registration/ICPPointAlign.hpp"
#include "lvr2/registration/EigenSVDPointAlign.hpp"
#include "lvr2/io/Timestamp.hpp"

#include <iomanip>
#include <chrono>

using namespace std;

namespace lvr2
{

ICPPointAlign::ICPPointAlign(SLAMScanPtr model, SLAMScanPtr data) :
    m_modelCloud(model), m_dataCloud(data)
{
    // Init default values
    m_maxDistanceMatch  = 25;
    m_maxIterations     = 50;
    m_epsilon           = 0.00001;
    m_verbose           = false;

    m_searchTree = KDTree::create(model, m_maxLeafSize);
}

Transformd ICPPointAlign::match()
{
    if (m_maxIterations == 0)
    {
        return Matrix4d::Identity();
    }

    auto start_time = chrono::steady_clock::now();

    double ret = 0.0, prev_ret = 0.0, prev_prev_ret = 0.0;
    EigenSVDPointAlign<double> align;
    int iteration = 0;

    Vector3d centroid_m = Vector3d::Zero();
    Vector3d centroid_d = Vector3d::Zero();
    Transformd transform = Matrix4d::Identity();
    Transformd delta = Matrix4d::Identity();

    size_t numPoints = m_dataCloud->numPoints();

    KDTree::Neighbor* neighbors = new KDTree::Neighbor[numPoints];

    for (iteration = 0; iteration < m_maxIterations; iteration++)
    {
        // Update break variables
        prev_prev_ret = prev_ret;
        prev_ret = ret;

        // Get point pairs
        size_t pairs = KDTree::nearestNeighbors(m_searchTree, m_dataCloud, neighbors, m_maxDistanceMatch, centroid_m, centroid_d);

        // Get transformation
        transform = Transformd::Identity();
        ret = align.alignPoints(m_dataCloud, neighbors, centroid_m, centroid_d, transform);

        // Apply transformation
        m_dataCloud->transform(transform, false);
        delta = delta * transform;

        if (m_verbose)
        {
            cout << timestamp << "ICP Error is " << ret << " in iteration " << iteration << " / " << m_maxIterations << " using " << pairs << " points." << endl;
        }

        // Check minimum distance
        if ((fabs(ret - prev_ret) < m_epsilon) && (fabs(ret - prev_prev_ret) < m_epsilon))
        {
            break;
        }
    }

    delete[] neighbors;

    auto duration = chrono::steady_clock::now() - start_time;
    cout << setw(6) << (int)(duration.count() / 1e6) << " ms, ";
    cout << "Error: " << fixed << setprecision(3) << setw(7) << ret;
    if (iteration < m_maxIterations)
    {
        cout << " after " << iteration << " Iterations";
    }
    cout << endl;
    if (m_verbose)
    {
        cout << "Result: " << endl << m_dataCloud->deltaPose() << endl;
    }

    return delta;
}

void ICPPointAlign::setMaxMatchDistance(double d)
{
    m_maxDistanceMatch = d;
}

void ICPPointAlign::setMaxIterations(int i)
{
    m_maxIterations = i;
}

void ICPPointAlign::setMaxLeafSize(int m)
{
    m_maxLeafSize = m;
}

void ICPPointAlign::setEpsilon(double e)
{
    m_epsilon = e;
}
void ICPPointAlign::setVerbose(bool verbose)
{
    m_verbose = verbose;
}

double ICPPointAlign::getMaxMatchDistance() const
{
    return m_maxDistanceMatch;
}

int ICPPointAlign::getMaxIterations() const
{
    return m_maxIterations;
}

int ICPPointAlign::getMaxLeafSize() const
{
    return m_maxLeafSize;
}

double ICPPointAlign::getEpsilon() const
{
    return m_epsilon;
}

bool ICPPointAlign::getVerbose() const
{
    return m_verbose;
}

} /* namespace lvr2 */
