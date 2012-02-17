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

template<typename VertexT, typename NormalT>
PCLKSurface<VertexT, NormalT>::PCLKSurface(PointBufferPtr loader, int kn, int kd)
    : PointsetSurface<VertexT>(loader)
{
    this->m_kn = kn;
    this->m_ki = 0;
    this->m_kd = kd;

    // Get buffer array and conver it to a PCL point cloud
    cout << timestamp << "Creating PCL point cloud" << endl;
    size_t numPoints;
    coord3fArr b_points = this->m_pointBuffer->getIndexedPointArray(numPoints);

    m_pointCloud  = pcl::PointCloud<pcl::PointXYZ>::Ptr (new pcl::PointCloud<pcl::PointXYZ>);
    m_pointCloud->resize(numPoints);
    float x, y, z;
    for(size_t i = 0; i < numPoints; i++)
    {
        x = b_points[i][0];
        y = b_points[i][1];
        z = b_points[i][2];

        m_pointCloud->points[i].x = x;
        m_pointCloud->points[i].y = y;
        m_pointCloud->points[i].z = z;
    }
    m_pointCloud->width = numPoints;
    m_pointCloud->height = 1;
}

template<typename VertexT, typename NormalT>
void PCLKSurface<VertexT, NormalT>::calculateSurfaceNormals()
{
    VertexT center = this->m_boundingBox.getCentroid();

    // Estimate normals
    cout << timestamp << "Estimating normals" << endl;

    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud (m_pointCloud);
    ne.setViewPoint(center.x, center.y, center.z);

    // Create an empty kdtree representation, and pass it to the normal estimation object.
    // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
#ifdef _PCL_VERSION_12_
    m_kdTree = pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr (new pcl::KdTreeFLANN<pcl::PointXYZ> ());
#else
    m_kdTree = pcl::search::KdTree<pcl::PointXYZ>::Ptr( new pcl::search::KdTree<pcl::PointXYZ> );
#endif
    ne.setSearchMethod (m_kdTree);

    // Output datasets
    m_pointNormals = pcl::PointCloud<pcl::Normal>::Ptr (new pcl::PointCloud<pcl::Normal>);
    ne.setKSearch(this->m_kn);

    // Compute the features
    ne.compute (*m_pointNormals);
    cout << timestamp << "Normal estimation done" << endl;
    cout << timestamp << "Saving normals" << endl;

    size_t numPoints = this->m_pointBuffer->getNumPoints();
    coord3fArr normals( new coord<float>[numPoints] );

    for(size_t i = 0; i < numPoints; i++)
    {
        normals[i][0] = m_pointNormals->points[i].normal[0];
        normals[i][1] = m_pointNormals->points[i].normal[1];
        normals[i][2] = m_pointNormals->points[i].normal[2];
    }

    this->m_pointBuffer->setIndexedPointNormalArray(normals, numPoints);
}

template<typename VertexT, typename NormalT>
VertexT PCLKSurface<VertexT, NormalT>::getInterpolatedNormal(VertexT position)
{

    std::vector< int > k_indices;
    std::vector< float > k_distances;

    pcl::PointXYZ qp;
    qp.x = position.x;
    qp.y = position.y;
    qp.z = position.z;

    // Query tree
    int res = m_kdTree->nearestKSearch(qp, this->m_kn, k_indices, k_distances);

	VertexT result(0,0,0);
	for (int i = 0; i < res; i++)
	{
	    int ind = k_indices[i];
		result[0] += m_pointNormals->points[ind].normal[0];
		result[1] += m_pointNormals->points[ind].normal[1];
		result[2] += m_pointNormals->points[ind].normal[2];
	}
	result /= res;
	return result;
}

template<typename VertexT, typename NormalT>
void PCLKSurface<VertexT, NormalT>::distance(VertexT v, float &projectedDistance, float &euklideanDistance)
{
    std::vector< int > k_indices;
    std::vector< float > k_distances;

    pcl::PointXYZ qp;
    qp.x = v.x;
    qp.y = v.y;
    qp.z = v.z;

    // Query tree
    int res = m_kdTree->nearestKSearch(qp, this->m_kd, k_indices, k_distances);

    // Round distance
    float q_distance = 0.0f;
    NormalT n;
    VertexT c;
    for(int i = 0; i < res; i++)
    {
        int ind = k_indices[i];
        NormalT tmp_n = NormalT(
                m_pointNormals->points[ind].normal[0],
                m_pointNormals->points[ind].normal[1],
                m_pointNormals->points[ind].normal[2]
                );

        VertexT tmp_p = VertexT(
                m_pointCloud->points[ind].x,
                m_pointCloud->points[ind].y,
                m_pointCloud->points[ind].z
                );


        //cout << ind << " / " << m_pointNormals->points.size() << " " << tmp << endl;
        n += tmp_n;
        c += tmp_p;

        q_distance += k_distances[i];
    }
    n /= res;
    c /= res;

    projectedDistance = (v - c) * n;
    euklideanDistance = (v - c).length();

}

} /* namespace lssr */
