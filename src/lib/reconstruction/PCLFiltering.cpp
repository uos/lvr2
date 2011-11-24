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

#include "PCLFiltering.hpp"

namespace lssr
{

PCLFiltering::PCLFiltering( PointBufferPtr loader )
{
    // TODO Auto-generated constructor stub
    m_pointCloud  = pcl::PointCloud<pcl::PointXYZ>::Ptr (new pcl::PointCloud<pcl::PointXYZ>);

    // Get data from loader object
    size_t numPoints;
    float** points = loader->getIndexedPointArray(numPoints);

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

        m_pointCloud->points[i].x = x;
        m_pointCloud->points[i].y = y;
        m_pointCloud->points[i].z = z;
    }

    // Create an empty kdtree representation, and pass it to the normal estimation object.
    // Its content will be filled inside the object, based on the given input dataset
    // (as no other search surface is given).
    m_kdTree = pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr (new pcl::KdTreeFLANN<pcl::PointXYZ> ());
    m_kdTree->setInputCloud (m_pointCloud);
}

void PCLFiltering::applyMLSProjection(float searchRadius)
{
    pcl::MovingLeastSquares<pcl::PointXYZ, pcl::Normal> mls;
    pcl::PointCloud<pcl::PointXYZ> mls_points;

    // Set Parameters
    mls.setInputCloud(m_pointCloud);
    mls.setPolynomialFit(true);
    mls.setSearchMethod(m_kdTree);
    mls.setSearchRadius(searchRadius);

    std::cout << timestamp << "Applying MSL projection" << std::endl;

    // Reconstruct
    mls.reconstruct(mls_points);

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
}

void PCLFiltering::applyOutlierRemoval(int meank, float thresh)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);

    std::cout << timestamp << "Applying outlier removal" << std::endl;

    // Create the filtering object
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud (m_pointCloud);
    sor.setMeanK (meank);
    sor.setStddevMulThresh (thresh);
    sor.filter (*cloud_filtered);

    std::cout << timestamp << "Filtered cloud has " << cloud_filtered->size() << " points" << std::endl;
    std::cout << timestamp << "Saving result" << std::endl;

    // Save filtered points
    m_pointCloud->resize(cloud_filtered->size());
    float x, y, z;
    for(size_t i = 0; i < cloud_filtered->size(); i++)
    {
        m_pointCloud->points[i].x = cloud_filtered->points[i].x;
        m_pointCloud->points[i].y = cloud_filtered->points[i].y;
        m_pointCloud->points[i].z = cloud_filtered->points[i].z;

    }
}

PointBufferPtr PCLFiltering::getPointBuffer()
{
    PointBufferPtr p( new PointBuffer );

    float* points = new float[3 * m_pointCloud->size()];
    for(int i = 0; i < m_pointCloud->size(); i++)
    {
        size_t pos = 3 * i;
        points[pos    ] = m_pointCloud->points[i].x;
        points[pos + 1] = m_pointCloud->points[i].y;
        points[pos + 2] = m_pointCloud->points[i].z;
    }

    p->setPointArray(points, m_pointCloud->size());
    return p;
}

PCLFiltering::~PCLFiltering()
{
    // TODO Auto-generated destructor stub
}

} /* namespace lssr */

#endif
