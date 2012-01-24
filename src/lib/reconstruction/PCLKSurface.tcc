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
 * PCLKSurface.cpp
 *
 *  @date 24.01.2012
 *  @author Thomas Wiemann
 */

#include "PCLKSurface.hpp"

namespace lssr
{

template<typename VertexT>
PCLKSurface<VertexT>::PCLKSurface(PointBufferPtr loader, int kn, int kd)
    : PointsetSurface(loader)
{
    this->m_kn = kn;
    this->m_ki = ki;
    this->m_kd = kd;

    m_pointCloud  = pcl::PointCloud<pcl::PointXYZ>::Ptr (new pcl::PointCloud<pcl::PointXYZ>);

    // Parse to PCL point cloud
    cout << timestamp << "Creating PCL point cloud" << endl;
    m_pointCloud->resize(this->m_numPoints);
    float x, y, z;
    for(size_t i = 0; i < this->m_numPoints; i++)
    {
        x = this->m_points[i][0];
        y = this->m_points[i][1];
        z = this->m_points[i][2];

        this->m_boundingBox.expand(x, y, z);

        m_pointCloud->points[i].x = x;
        m_pointCloud->points[i].y = y;
        m_pointCloud->points[i].z = z;
    }
    m_pointCloud->width = this->m_numPoints;
    m_pointCloud->height = 1;
}

template<typename VertexT>
void PCLKSurface<VertexT>::calculateSurfaceNormals()
{
    VertexT center = this->m_boundingBox.getCentroid();

    // Estimate normals
    cout << timestamp << "Estimating normals" << endl;

    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud (m_pointCloud);
    ne.setViewPoint(center.x, center.y, center.z);

    // Create an empty kdtree representation, and pass it to the normal estimation object.
    // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
    m_kdTree = pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr (new pcl::KdTreeFLANN<pcl::PointXYZ> ());
    ne.setSearchMethod (m_kdTree);

    // Output datasets
    m_pointNormals = pcl::PointCloud<pcl::Normal>::Ptr (new pcl::PointCloud<pcl::Normal>);


    ne.setKSearch(this->m_kn);

    // Compute the features
    ne.compute (*m_pointNormals);
    cout << timestamp << "Normal estimation done" << endl;
}

template<typename VertexT>
PointBufferPtr PCLKSurface<VertexT>::pointBuffer()
{
    coord3fArr normals( new coord<float>[this->m_numPoints] );

    for(size_t i = 0; i < this->m_numPoints; i++)
    {
        normals[i][0] = m_pointNormals->points[i].normal[0];
        normals[i][1] = m_pointNormals->points[i].normal[1];
        normals[i][2] = m_pointNormals->points[i].normal[2];
    }

    this->m_pointBuffer->setIndexedPointNormalArray(normals, this->m_numPoints);
    return this->m_pointBuffer;
}

} /* namespace lssr */
