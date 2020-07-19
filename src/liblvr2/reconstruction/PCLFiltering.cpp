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
 * PCLFiltering.cpp
 *
 *  @date 22.11.2011
 *  @author Thomas Wiemann
 */

#ifdef LVR2_USE_PCL

#include "lvr2/reconstruction/PCLFiltering.hpp"
#include <pcl/pcl_config.h>

namespace lvr2
{


PCLFiltering::PCLFiltering( PointBufferPtr loader )
{

    // Check if we have RGB data
    size_t numPoints = loader->numPoints();
    m_useColors = loader->hasColors();

    m_pointCloud  = pcl::PointCloud<pcl::PointXYZRGB>::Ptr (new pcl::PointCloud<pcl::PointXYZRGB>);


    // Get data from loader object
    FloatChannelOptional points = loader->getFloatChannel("points");
    UCharChannelOptional colors = loader->getUCharChannel("colors");

    size_t numColors = 0;
    if(colors)
    {
        numColors = (*colors).numElements();
    }

    if(m_useColors)
    {
        assert(numColors == numPoints);
    }


    // Parse to PCL point cloud
    std::cout << timestamp << "Creating PCL point cloud for filtering" << std::endl;
    std::cout << timestamp << "Point cloud has " << numPoints << " points" << std::endl;
    m_pointCloud->resize(numPoints);
    float x, y, z;
    for(size_t i = 0; i < numPoints; i++)
    {
        x = (*points)[i][0];
        y = (*points)[i][1];
        z = (*points)[i][2];

        if(m_useColors)
        {
//            // Pack color information
            uint8_t r = (*colors)[i][0];
            uint8_t g = (*colors)[i][1];
            uint8_t b = (*colors)[i][2];

//            uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
//            m_pointCloud->points[i].rgb = *reinterpret_cast<float*>(&rgb);

            m_pointCloud->points[i] = pcl::PointXYZRGB(r,g,b);
        }

        m_pointCloud->points[i].x = x;
        m_pointCloud->points[i].y = y;
        m_pointCloud->points[i].z = z;
    }

    m_pointCloud->width = numPoints;
    m_pointCloud->height = 1;

    // Create an empty kdtree representation, and pass it to the normal estimation object.
    // Its content will be filled inside the object, based on the given input dataset
    // (as no other search surface is given).
	m_kdTree = pcl::search::KdTree<pcl::PointXYZRGB>::Ptr( new pcl::search::KdTree<pcl::PointXYZRGB> );
    m_kdTree->setInputCloud (m_pointCloud);
}

void PCLFiltering::applyMLSProjection(float searchRadius)
{
    pcl::MovingLeastSquares<pcl::PointXYZRGB, pcl::PointNormal> mls;
    pcl::PointCloud<pcl::PointNormal> mls_points;

    // Set Parameters
    mls.setInputCloud(m_pointCloud);
    mls.setPolynomialFit(true);
    mls.setSearchMethod(m_kdTree);
    mls.setSearchRadius(searchRadius);

    std::cout << timestamp << "Applying MSL projection" << std::endl;

    // Reconstruct
    mls.process(mls_points);

    std::cout << timestamp << "Filtered cloud has " << mls_points.size() << " points" << std::endl;
    std::cout << timestamp << "Saving result" << std::endl;

    // Save filtered points
    m_pointCloud->resize(mls_points.size());
    float x, y, z;
    for(size_t i = 0; i < mls_points.size(); i++)
    {
        m_pointCloud->points[i].x = mls_points.points[i].x;
        m_pointCloud->points[i].y = mls_points.points[i].y;
        m_pointCloud->points[i].z = mls_points.points[i].z;
    }

    m_pointCloud->height = 1;
    m_pointCloud->width = mls_points.size();

}

void PCLFiltering::applyOutlierRemoval(int meank, float thresh)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGB>);

    std::cout << timestamp << "Applying outlier removal" << std::endl;

    // Create the filtering object
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
    sor.setInputCloud (m_pointCloud);
    sor.setMeanK (meank);
    sor.setStddevMulThresh (thresh);
    sor.filter (*cloud_filtered);

    std::cout << timestamp << "Filtered cloud has " << cloud_filtered->size() << " points" << std::endl;
    std::cout << timestamp << "Saving result" << std::endl;

    m_pointCloud->width = cloud_filtered->width;
    m_pointCloud->height = cloud_filtered->height;
    m_pointCloud->points.swap (cloud_filtered->points);
}

PointBufferPtr PCLFiltering::getPointBuffer()
{
    PointBufferPtr p( new PointBuffer );

    floatArr points(new float[3 * m_pointCloud->size()]);
    ucharArr colors;

    if(m_useColors)
    {
       colors = ucharArr(new unsigned char[3 * m_pointCloud->size()]);
    }

    for(int i = 0; i < m_pointCloud->size(); i++)
    {
        size_t pos = 3 * i;
        points[pos    ] = m_pointCloud->points[i].x;
        points[pos + 1] = m_pointCloud->points[i].y;
        points[pos + 2] = m_pointCloud->points[i].z;

        if(m_useColors)
        {
            colors[pos    ] = m_pointCloud->points[i].r;
            colors[pos + 1] = m_pointCloud->points[i].g;
            colors[pos + 2] = m_pointCloud->points[i].b;
//            std::cout << (int)m_pointCloud->points[i].r << " "
//                 <<  (int)m_pointCloud->points[i].g << " "
//                 <<  (int)m_pointCloud->points[i].b << " " << std::endl;
        }
    }

    p->setPointArray(points, m_pointCloud->size());

    if(m_useColors)
    {
        p->setColorArray(colors, m_pointCloud->size());
    }

    return p;
}

PCLFiltering::~PCLFiltering()
{
    // TODO Auto-generated destructor stub
}

} /* namespace lvr2 */

#endif
