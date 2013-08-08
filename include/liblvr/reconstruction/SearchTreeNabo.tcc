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
 * SearchTreeNabo.tcc 
 *
 *  Created on: 02.01.2012
 *      Author: Florian Otte
 */

// stl includes
#include <limits>

// External libraries in lvr source tree
#include "../Eigen/Dense"

// boost libraries
#include <boost/filesystem.hpp>

// lvr includes

using std::cout;
using std::endl;
using std::numeric_limits;

namespace lvr {

template<typename VertexT>
SearchTreeNabo< VertexT >::SearchTreeNabo(PointBufferPtr buffer , long unsigned int &n_points, const int &kn, const int &ki, const int &kd, const bool &useRansac )
{
    // Store parameters
    this->m_ki = ki;
    this->m_kn = kn;
    this->m_kd = kd;

    size_t n;
    coord3fArr points = buffer->getIndexedPointArray(n);

    m_points = Eigen::MatrixXf(n_points, 3);
    for( size_t i(0); i < n_points; ++i )
    {
        m_points(i, 0) = points[i].x;
        m_points(i, 1) = points[i].y;
        m_points(i, 2) = points[i].z;
    }

    // Create Nabo Kd-tree
    enum Nabo::NearestNeighbourSearch<float>::SearchType         seType = Nabo::NearestNeighbourSearch<float>::KDTREE_TREE_HEAP;
    cout << timestamp << "Creating NABO Kd-Tree" << endl;
    m_pointTree = Nabo::NNSearchF::create( m_points, 3, seType );
}


template<typename VertexT>
SearchTreeNabo< VertexT >::~SearchTreeNabo() {
    if( m_pointTree )
        delete m_pointTree;
}


template<typename VertexT>
void SearchTreeNabo< VertexT >::kSearch( coord< float > &qp, int neighbours, vector< ulong > &indices, vector< double > &distances )
{
    Eigen::Vector3f q;
    q[0] = qp.x;
    q[1] = qp.y;
    q[2] = qp.z;

    Eigen::VectorXi ind( neighbours );
    Eigen::VectorXf dist( neighbours );

    enum Nabo::NearestNeighbourSearch<float>::SearchOptionFlags opType = Nabo::NearestNeighbourSearch<float>::SORT_RESULTS;
    m_pointTree->knn( q, ind, dist, neighbours, 0,opType);

    //      cout << "After Search:\n\tind.cols(): " << ind.cols() << "\n\tind.rows(): " << ind.rows() << "\n\n\tdist.cols(): " << dist.cols() << "\n\tdist.rows(): " << dist.rows() << endl;

    // store the found data
    indices.resize( ind.rows() );
    distances.resize( dist.rows() );
    int valid = 0;
    for( size_t i(0); i < ind.rows(); i++ )
    {
        if( !isinf( dist(i, 0) ) && !isnan( dist(i, 0) ) )
        {
            indices[i] = ind(i, 0);
            distances[i] = dist(i, 0);
            valid++;
        }
    }
    indices.resize( valid );
    distances.resize( valid );
}


/*
   Begin of radiusSearch implementations
 */
template<typename VertexT>
void SearchTreeNabo< VertexT >::radiusSearch( float qp[3], double r, vector< ulong > &indices )
{
    // clear possibly old information
    indices.clear();
    // keep track of found distances and indices
    vector< double > distances;

    double squared_radius = r*r;
    double max_radius = numeric_limits< double >::min();
    int k = 10;
    while( max_radius < r ){
        SearchTree< VertexT >::kSearch( qp, k, indices, distances );

        // check distances for all neighbours
        for( int i=0; i < distances.size(); i++ )
        {
            max_radius = (max_radius > distances[i]) ? max_radius : distances[i];
            if( distances[i] < r )
            {
                indices.push_back( indices[i] );
            }
        }
        k *= 2;
    }
}


template<typename VertexT>
void SearchTreeNabo< VertexT >::radiusSearch( VertexT& qp, double r, vector< ulong > &indices )
{
    float qp_arr[3];
    qp_arr[0] = qp[0];
    qp_arr[1] = qp[1];
    qp_arr[2] = qp[2];
    this->radiusSearch( qp_arr, r, indices );
}


template<typename VertexT>
void SearchTreeNabo< VertexT >::radiusSearch( const VertexT& qp, double r, vector< ulong > &indices )
{
    float qp_arr[3];
    qp_arr[0] = qp[0];
    qp_arr[1] = qp[1];
    qp_arr[2] = qp[2];
    this->radiusSearch( qp_arr, r, indices );
}


template<typename VertexT>
void SearchTreeNabo< VertexT >::radiusSearch( coord< float >& qp, double r, vector< ulong > &indices )
{
    float qp_arr[3];
    qp_arr[0] = qp[0];
    qp_arr[1] = qp[1];
    qp_arr[2] = qp[2];
    this->radiusSearch( qp_arr, r, indices );
}


template<typename VertexT>
void SearchTreeNabo< VertexT >::radiusSearch( const coord< float >& qp, double r, vector< ulong > &indices )
{
    float qp_arr[3];
    coord< float > qpcpy = qp;
    qp_arr[0] = qpcpy[0];
    qp_arr[1] = qpcpy[1];
    qp_arr[2] = qpcpy[2];
    this->radiusSearch( qp_arr, r, indices );
}


} // namespace lvr
