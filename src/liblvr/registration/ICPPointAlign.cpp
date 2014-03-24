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
 * ICPPointAlign.cpp
 *
 *  @date Mar 18, 2014
 *  @author Thomas Wiemann
 */
#include "registration/ICPPointAlign.hpp"
#include "registration/EigenSVDPointAlign.hpp"
#include "io/Timestamp.hpp"

#include <fstream>
using std::ofstream;

namespace lvr
{

ICPPointAlign::ICPPointAlign(PointBufferPtr model, PointBufferPtr data, Matrix4f transform) :
    m_modelCloud(model), m_dataCloud(data), m_transformation(transform)
{
    // Init default values
    m_epsilon               = 0.00001;
    m_maxDistanceMatch      = 25;
    m_maxIterations         = 50;

    size_t numPoints = model->getNumPoints();

    // Create search tree
    m_searchTree = SearchTreeFlann<Vertexf>::Ptr(new SearchTreeFlann<Vertexf>(model, numPoints));
}

Matrix4f ICPPointAlign::match()
{
    if(m_maxIterations == 0)
    {
        return Matrix4f();
    }

    // Quick hack
    ofstream pose1("scan001.frames");
    ofstream pose2("scan002.frames");

    double ret = 0.0, prev_ret = 0.0, prev_prev_ret = 0.0;
    EigenSVDPointAlign align;
    for(int i = 0; i < m_maxIterations; i++)
    {
        // Update break variables
        prev_prev_ret = prev_ret;
        prev_ret = ret;

        // Get point pairs
        Vertexf         centroid_m;
        Vertexf         centroid_d;
        Matrix4f        transform;
        double          sum;

        cout << i << endl;
        cout << m_transformation;
        cout << endl;

        PointPairVector pairs;
        getPointPairs(pairs, centroid_m, centroid_d, sum);

        // Get transformation (if possible)
        ret = align.alignPoints(pairs, centroid_m, centroid_d, transform);

        // Apply transformation
        m_transformation *= transform;

        cout << timestamp << "ICP Error is " << ret << " in iteration " << i << " / " << m_maxIterations << " using " << pairs.size() << " points."<< endl;

        pose1 << "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 2" << endl;

        for(int j = 0; j < 16; j++)
        {
            pose2 << m_transformation[j] << " ";
        }
        pose2 << "1" << endl;

        // Check minimum distance
        if ((fabs(ret - prev_ret) < m_epsilon) && (fabs(ret - prev_prev_ret) < m_epsilon))
        {
            break;
        }
    }
    return m_transformation;
}

void ICPPointAlign::getPointPairs(PointPairVector& pairs, Vertexf& centroid_m, Vertexf& centroid_d, double& sum)
{
    size_t n;
    bool ok;
    Matrix4f transformInv = m_transformation.inv(ok);
    floatArr dataPoints = m_dataCloud->getPointArray(n);
    sum = 0;

    #pragma omp parallel
    {
        PointPairVector privatePairs;
        Vertexf centroid_mP;
        Vertexf centroid_dP;
        vector<Vertexf> neighbors;

        #pragma omp for nowait //fill vec_private in parallel
        for(int i = 0; i < m_dataCloud->getNumPoints(); i++)
        {
            // Get vertex representation of current data point
            Vertexf dataPoint(dataPoints[i * 3], dataPoints[i * 3 + 1], dataPoints[i * 3 + 2]);
            Vertexf queryPoint = m_transformation * dataPoint;

            // Perform inverse transformation on query point to
            // it's position relative to the model point data
            Vertexf inverseQueryPoint = transformInv * dataPoint;

            // Get closest point to "inverse query point"
            neighbors.clear();
            m_searchTree->kSearch(inverseQueryPoint, 1, neighbors);

            // If closest point was found, transform back to the corresponding
            // position in the data reference frame
            if(neighbors.size())
            {
                Vertexf closest = m_transformation * neighbors[0];
                if( (closest - queryPoint).length2() < m_maxDistanceMatch)
                {
                    centroid_dP += queryPoint;
                    centroid_mP += closest;

                    sum += (closest - queryPoint).length2();

                    std::pair<Vertexf, Vertexf> ptPair(closest, queryPoint);
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

} /* namespace lvr */
