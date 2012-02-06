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


 /*
 * SearchTreeFlann.tcc
 *
 *  Created on: 02.01.2012
 *      Author: Florian Otte
 */

// stl includes
#include <limits>

// External libraries in lssr source tree
#include "../Eigen/Dense"

// boost libraries
#include <boost/filesystem.hpp>

// lssr includes

using std::cout;
using std::endl;
using std::numeric_limits;

using pcl::PointCloud;
using pcl::PointXYZ;
using pcl::KdTreeFLANN;

namespace lssr {

template<typename VertexT>
SearchTreeFlann< VertexT >::SearchTreeFlann( PointBufferPtr buffer, long unsigned int &n_points, const int &kn, const int &ki, const int &kd )
{
    // Store parameters
    this->m_ki = ki;
    this->m_kn = kn;
    this->m_kd = kd;

    size_t n_colors;
    coord3fArr points = buffer->getIndexedPointArray(n_points);
    color3bArr colors = buffer->getIndexedPointColorArray(n_colors);

    // initialize pointCloud for pcl.
    m_pointCloud = pcl::PointCloud< pcl::PointXYZRGB >::Ptr( new pcl::PointCloud< pcl::PointXYZRGB >() );
    m_pointCloud->resize( n_points );

    cout << n_colors << endl;

    // Store points in pclCloud
    for( int i(0); i < n_points; ++i )
    {
        m_pointCloud->points[i].x = points[i].x;
        m_pointCloud->points[i].y = points[i].y;
        m_pointCloud->points[i].z = points[i].z;

        // Assign color data
        if(n_colors)
        {
            m_pointCloud->points[i].r = colors[i].r;
            m_pointCloud->points[i].g = colors[i].g;
            m_pointCloud->points[i].b = colors[i].b;
        }
        else
        {
            m_pointCloud->points[i].r = 0.0;
            m_pointCloud->points[i].g = 1.0;
            m_pointCloud->points[i].b = 0.0;
        }
    }

    // Set pointCloud dimensions
    m_pointCloud->width  = n_points;
    m_pointCloud->height = 1;

    // initialize kd-Tree
    cout << timestamp << "Initialising PCL<Flann> Kd-Tree" << endl;
    m_kdTree = KdTreeFLANN< pcl::PointXYZRGB >::Ptr( new KdTreeFLANN< pcl::PointXYZRGB >() );
    m_kdTree->setInputCloud(m_pointCloud);
}


template<typename VertexT>
SearchTreeFlann< VertexT >::~SearchTreeFlann() {
}


template<typename VertexT>
void SearchTreeFlann< VertexT >::kSearch( coord< float > &qp, int neighbours, vector< ulong > &indices, vector< double > &distances )
{
    // get pcl compatible point.
    pcl::PointXYZRGB pcl_qp;
    pcl_qp.x = qp[0];
    pcl_qp.y = qp[1];
    pcl_qp.z = qp[2];

    // get pcl-compatible indice and distance vectors
    vector< int > ind;
    vector< float > dist;

    // perform the search
    m_kdTree->nearestKSearch( pcl_qp, neighbours, ind, dist );

    // copy information to interface conform vector types
    indices.clear();
    distances.clear();
    indices.insert( indices.begin(), ind.begin(), ind.end() );
    distances.insert( distances.begin(), dist.begin(), dist.end() );

}

template<typename VertexT>
void SearchTreeFlann< VertexT >::kSearch(VertexT qp, int k, vector< VertexT > &neighbors)
{
    vector<ulong> indices;
    SearchTree<VertexT>::kSearch(qp, k, indices);
    for(size_t i = 0; i < indices.size(); i++)
    {
        neighbors.push_back(
                VertexT(m_pointCloud->points[indices[i]].x,
                        m_pointCloud->points[indices[i]].y,
                        m_pointCloud->points[indices[i]].z,
                        m_pointCloud->points[indices[i]].r,
                        m_pointCloud->points[indices[i]].g,
                        m_pointCloud->points[indices[i]].b));
    }
}

/*
   Begin of radiusSearch implementations
 */
template<typename VertexT>
void SearchTreeFlann< VertexT >::radiusSearch( float qp[3], double r, vector< ulong > &indices )
{
    // TODO: Implement me!
}


template<typename VertexT>
void SearchTreeFlann< VertexT >::radiusSearch( VertexT& qp, double r, vector< ulong > &indices )
{
    float qp_arr[3];
    qp_arr[0] = qp[0];
    qp_arr[1] = qp[1];
    qp_arr[2] = qp[2];
    this->radiusSearch( qp_arr, r, indices );
}


template<typename VertexT>
void SearchTreeFlann< VertexT >::radiusSearch( const VertexT& qp, double r, vector< ulong > &indices )
{
    float qp_arr[3];
    qp_arr[0] = qp[0];
    qp_arr[1] = qp[1];
    qp_arr[2] = qp[2];
    this->radiusSearch( qp_arr, r, indices );
}


template<typename VertexT>
void SearchTreeFlann< VertexT >::radiusSearch( coord< float >& qp, double r, vector< ulong > &indices )
{
    float qp_arr[3];
    qp_arr[0] = qp[0];
    qp_arr[1] = qp[1];
    qp_arr[2] = qp[2];
    this->radiusSearch( qp_arr, r, indices );
}


template<typename VertexT>
void SearchTreeFlann< VertexT >::radiusSearch( const coord< float >& qp, double r, vector< ulong > &indices )
{
    float qp_arr[3];
    coord< float > qpcpy = qp;
    qp_arr[0] = qpcpy[0];
    qp_arr[1] = qpcpy[1];
    qp_arr[2] = qpcpy[2];
    this->radiusSearch( qp_arr, r, indices );
}
} // namespace lssr
