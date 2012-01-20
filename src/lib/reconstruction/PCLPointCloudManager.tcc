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
 * PCLPointCloudManager.cpp
 *
 *  @date 08.08.2011
 *  @author Thomas Wiemann
 */

#include "PCLPointCloudManager.hpp"
#include "io/Timestamp.hpp"

#include <vector>

namespace lssr
{

template<typename VertexT, typename NormalT>
PCLPointCloudManager<VertexT, NormalT>::PCLPointCloudManager( PointBufferPtr loader, int kn, int ki, int kd )
    : PointCloudManager<VertexT, NormalT>(loader)
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

template<typename VertexT, typename NormalT>
void PCLPointCloudManager<VertexT, NormalT>::calcNormals()
{
    VertexT center = this->m_boundingBox.getCentroid();

    // Estimate normals
    cout << timestamp << "Estimating normals" << endl;

    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud (m_pointCloud);
    ne.setViewPoint(center.x, center.y, center.z);

    // Create an empty kdtree representation, and pass it to the normal
    // estimation object. Its content will be filled inside the object,
    // based on the given input dataset (as no other search surface is given).
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
}

template<typename VertexT, typename NormalT>
PointBufferPtr PCLPointCloudManager<VertexT, NormalT>::pointBuffer()
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

template<typename VertexT, typename NormalT>
PCLPointCloudManager<VertexT, NormalT>::~PCLPointCloudManager()
{
    // TODO Auto-generated destructor stub
}

template<typename VertexT, typename NormalT>
void PCLPointCloudManager<VertexT, NormalT>::getkClosestVertices(const VertexT &v,
         const size_t &k, vector<VertexT> &nb)
{
    std::vector< int > k_indices;
    std::vector< float > k_distances;

    pcl::PointXYZ qp;
    qp.x = v.x;
    qp.y = v.y;
    qp.z = v.z;

    // Query tree
    size_t res = m_kdTree->nearestKSearch(qp, k, k_indices, k_distances);

    // Check number of found neighbours
    if(res != k)
    {
        cout << timestamp << "PCLPointCloudManager::getkClosestVertices() : Warning: Number of found neighbours != k_n." << endl;
    }

    // Parse result
    for(size_t i = 0; i < res; i++)
    {
        int index = k_indices[i];
        if(this->m_colors != 0)
        	nb.push_back(VertexT(m_pointCloud->points[index].x,
                             m_pointCloud->points[index].y,
                             m_pointCloud->points[index].z,
                             this->m_colors[index][0],
                             this->m_colors[index][1],
                             this->m_colors[index][2]));
        else
        	nb.push_back(VertexT(m_pointCloud->points[index].x,
        	                             m_pointCloud->points[index].y,
        	                             m_pointCloud->points[index].z));
    }
}


template<typename VertexT, typename NormalT>
void PCLPointCloudManager<VertexT, NormalT>::getkClosestNormals(const VertexT &n,
         const size_t &k, vector<NormalT> &nb)
{
      std::vector< int > k_indices;
      std::vector< float > k_distances;

      pcl::PointXYZ qp;
      qp.x = n.x;
      qp.y = n.y;
      qp.z = n.z;

      // Query tree
      size_t res = m_kdTree->nearestKSearch(qp, k, k_indices, k_distances);

      // Check number of found neighbours
      if(res != k)
      {
          cout << timestamp << "PCLPointCloudManager::getkClosestNormals() : Warning: Number of found neighbours != k_n." << endl;
      }

      // Parse result
      for(size_t i = 0; i < res; i++)
      {
          int index = k_indices[i];
          nb.push_back(NormalT(m_pointNormals->points[index].data_c[0],
                               m_pointNormals->points[index].data_c[1],
                               m_pointNormals->points[index].data_c[2]));
      }
}

template<typename VertexT, typename NormalT>
void PCLPointCloudManager<VertexT, NormalT>::distance(VertexT v, float &projectedDistance, float &euklideanDistance)
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


template<typename VertexT, typename NormalT>
void PCLPointCloudManager<VertexT, NormalT>::radiusSearch(const VertexT &v, double r, vector<VertexT> &resV, vector<NormalT> &resN)
{
    std::vector< int > result_indices;
    std::vector< float > result_distances;

    pcl::PointXYZ qp;
    qp.x = v.x;
    qp.y = v.y;
    qp.z = v.z;

    size_t res = m_kdTree->radiusSearch(qp, r, result_indices, result_distances);
    // Parse result
    for(size_t i = 0; i < res; i++)
    {
    	int index = result_indices[i];
    	resV.push_back(VertexT(m_pointCloud->points[index].x,
    			m_pointCloud->points[index].y,
    			m_pointCloud->points[index].z));
    	resN.push_back(NormalT(m_pointNormals->points[index].normal[0],
    			m_pointNormals->points[index].normal[1],
    			m_pointNormals->points[index].normal[2]));
    }
}

} // namespace lssr
