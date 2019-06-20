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
#include "lvr2/reconstruction/SearchTreeFlann.hpp"


#include <fstream>
using std::ofstream;

namespace lvr2
{

template <typename BaseVecT>
ICPPointAlign<BaseVecT>::ICPPointAlign(PointBufferPtr model, PointBufferPtr data, Matrix4<BaseVecT> transform) :
    m_modelCloud(model), m_transformation(transform)
{
    // Init default values
    m_epsilon               = 0.00001;
    m_maxDistanceMatch      = 25;
    m_maxIterations         = 50;

    size_t numPoints = data->numPoints();

    // Transform data points according to initial pose estimation
    m_dataCloud = PointBufferPtr(new PointBuffer);
    size_t n = numPoints;
    floatArr o_points = data->getPointArray();
    floatArr t_points(new float[3 * n]);

    for(size_t i = 0; i < numPoints; i++)
    {
        BaseVecT v(o_points[3 * i], o_points[3 * i + 1], o_points[3 * i + 2]);
        BaseVecT t  = transform * v;
        t_points[3 * i    ] = t[0];
        t_points[3 * i + 1] = t[1];
        t_points[3 * i + 2] = t[2];
    }
    m_dataCloud->setPointArray(t_points, n);

    // Create search tree
    m_searchTree = SearchTreePtr<BaseVecT>(new SearchTreeFlann<BaseVecT>(model));

}

template <typename BaseVecT>
Matrix4<BaseVecT> ICPPointAlign<BaseVecT>::match()
{
    if(m_maxIterations == 0)
    {
        return Matrix4<BaseVecT>();
    }

    double ret = 0.0, prev_ret = 0.0, prev_prev_ret = 0.0;
    EigenSVDPointAlign<BaseVecT> align;
    for(int i = 0; i < m_maxIterations; i++)
    {
        // Update break variables
        prev_prev_ret = prev_ret;
        prev_ret = ret;

        // Get point pairs
        BaseVecT  centroid_m;
        BaseVecT  centroid_d;
        Matrix4<BaseVecT> transform;
        double            sum;


        PointPairVector<BaseVecT> pairs;
        getPointPairs(pairs, centroid_m, centroid_d, sum);

        // Get transformation (if possible)
        ret = align.alignPoints(pairs, centroid_m, centroid_d, transform);

        cout << timestamp << "CORRECTION" << endl;
        cout << transform << endl;

        // Apply transformation
        m_transformation *= transform;

        cout << timestamp << "TRANSFORMATION: " << endl;
        cout << m_transformation << endl;

        cout << timestamp << "ICP Error is " << ret << " in iteration " << i << " / " << m_maxIterations << " using " << pairs.size() << " points."<< endl;
        //cout << m_transformation << endl;
        // Check minimum distance
        if ((fabs(ret - prev_ret) < m_epsilon) && (fabs(ret - prev_prev_ret) < m_epsilon))
        {
			cout << timestamp << " Error below m_epsilon " << endl;
            break;
        }
    }
    return m_transformation;
}

template <typename BaseVecT>
void ICPPointAlign<BaseVecT>::getPointPairs(PointPairVector<BaseVecT>& pairs, BaseVecT& centroid_m, BaseVecT& centroid_d, double& sum)
{
    size_t n = m_dataCloud->numPoints();
    bool ok;
    Matrix4<BaseVecT> transformInv = m_transformation.inv(ok);
    floatArr dataPoints = m_dataCloud->getPointArray();
    sum = 0;

    #pragma omp parallel
    {
        PointPairVector<BaseVecT> privatePairs;
        BaseVecT centroid_mP;
        BaseVecT centroid_dP;
        vector<size_t> neighbors;

        #pragma omp for nowait //fill vec_private in parallel
        for(size_t i = 0; i < m_dataCloud->numPoints(); i++)
        {
            // Get vertex representation of current data point
            BaseVecT t(dataPoints[i * 3], dataPoints[i * 3 + 1], dataPoints[i * 3 + 2]);


            // Perform inverse transformation on query point to
            // it's position relative to the model point data
            BaseVecT s = transformInv * t;

            // Get closest point to "inverse query point"
            neighbors.clear();
            m_searchTree->kSearch(s, 1, neighbors);

            // If closest point was found, transform back to the corresponding
            // position in the data reference frame
            if(neighbors.size())
            {
                FloatChannelOptional pts_channel = m_modelCloud->getFloatChannel("points");
                FloatChannel pts = *pts_channel;
                BaseVecT nb_pt = pts[neighbors[0]];
                BaseVecT closest = m_transformation * nb_pt;
                if( (closest - t).length() < m_maxDistanceMatch)
                {
                    centroid_dP += closest;
                    centroid_mP += t;

                    sum += (closest - t).length2();

                    std::pair<BaseVecT, BaseVecT> ptPair(t, closest);
                    privatePairs.push_back(ptPair);
                }
            }
        }
        #pragma omp critical
        pairs.insert(pairs.end(), privatePairs.begin(), privatePairs.end());

        if(pairs.size())
        {
            centroid_d += centroid_dP;
            centroid_m += centroid_mP;
        }
        else
        {
            cout << timestamp << "Warning: ICPPointAlign::getPointPairs(): No correspondences found." << endl;
        }
    }
    centroid_m /= pairs.size();
    centroid_d /= pairs.size();
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
