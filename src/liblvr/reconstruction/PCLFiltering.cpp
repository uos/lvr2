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
 * PCLFiltering.cpp
 *
 *  @date 22.11.2011
 *  @author Thomas Wiemann
 */

#ifdef _USE_PCL_

#include "reconstruction/PCLFiltering.hpp"
#include <pcl/pcl_config.h>

namespace lvr
{


PCLFiltering::PCLFiltering( PointBufferPtr loader )
{

    // Check if we have RGB data
    size_t numColors;
    m_useColors = bool(loader->getPointColorArray(numColors));

    m_pointCloud  = pcl::PointCloud<pcl::PointXYZRGB>::Ptr (new pcl::PointCloud<pcl::PointXYZRGB>);


    // Get data from loader object
    size_t numPoints;
    coord3fArr points = loader->getIndexedPointArray(numPoints);
    color3bArr colors;

    if(m_useColors)
    {
        assert(numColors == numPoints);
        colors = loader->getIndexedPointColorArray(numColors);
    }


    // Parse to PCL point cloud
    std::cout << timestamp << "Creating PCL point cloud for filtering" << std::endl;
    std::cout << timestamp << "Point cloud has " << numPoints << " points" << std::endl;
    m_pointCloud->resize(numPoints);
    float x, y, z;
    for(size_t i = 0; i < numPoints; i++)
    {
        x = points[i][0];
        y = points[i][1];
        z = points[i][2];

        if(m_useColors)
        {
//            // Pack color information
            uint8_t r = colors[i][0];
            uint8_t g = colors[i][1];
            uint8_t b = colors[i][2];

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
#ifdef _PCL_VERSION_12_
    m_kdTree = pcl::KdTreeFLANN<pcl::PointXYZRGB>::Ptr (new pcl::KdTreeFLANN<pcl::PointXYZRGB> ());
#else
	m_kdTree = pcl::search::KdTree<pcl::PointXYZRGB>::Ptr( new pcl::search::KdTree<pcl::PointXYZRGB> );
#endif
     
    m_kdTree->setInputCloud (m_pointCloud);
}

void PCLFiltering::applyMLSProjection(float searchRadius)
{
#if defined PCL_MAJOR_VERSION && defined PCL_MINOR_VERSION && PCL_MAJOR_VERSION == 1 && PCL_MINOR_VERSION >= 6
    pcl::MovingLeastSquares<pcl::PointXYZRGB, pcl::PointNormal> mls;
    pcl::PointCloud<pcl::PointNormal> mls_points;
#else
    pcl::MovingLeastSquares<pcl::PointXYZRGB, pcl::Normal> mls;
    pcl::PointCloud<pcl::PointXYZRGB> mls_points;
#endif

    // Set Parameters
    mls.setInputCloud(m_pointCloud);
    mls.setPolynomialFit(true);
    mls.setSearchMethod(m_kdTree);
    mls.setSearchRadius(searchRadius);

    std::cout << timestamp << "Applying MSL projection" << std::endl;

    // Reconstruct
#if defined PCL_MAJOR_VERSION && defined PCL_MINOR_VERSION && PCL_MAJOR_VERSION == 1 && PCL_MINOR_VERSION >= 6
    mls.process(mls_points);
#else
    mls.reconstruct(mls_points);
#endif

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
        p->setPointColorArray(colors, m_pointCloud->size());
    }

    return p;
}

PCLFiltering::~PCLFiltering()
{
    // TODO Auto-generated destructor stub
}

} /* namespace lvr */

#endif
