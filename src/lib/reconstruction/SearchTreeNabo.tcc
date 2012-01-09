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

// External libraries in lssr source tree
#include "../Eigen/Dense"

// boost libraries
#include <boost/filesystem.hpp>

// lssr includes

using std::cout;
using std::endl;
using std::numeric_limits;

namespace lssr {

template<typename VertexT, typename NormalT>
   SearchTreeNabo< VertexT, NormalT >::SearchTreeNabo( coord3fArr points, long unsigned int &n_points, const int &kn, const int &ki, const int &kd, const bool &useRansac ) 
   {
      // Store parameters
      this->m_ki = ki;
      this->m_kn = kn;
      this->m_kd = kd;

      m_points = Eigen::MatrixXf(n_points, 3);

      for( long unsigned int i(0); i < n_points; ++i )
      {
         m_points(i, 0) = points[i].x;
         m_points(i, 1) = points[i].y;
         m_points(i, 2) = points[i].z;
      }

      Eigen::MatrixXf M = Eigen::MatrixXf::Random(3, 100);

      // Create Nabo Kd-tree
      cout << timestamp << "Creating NABO Kd-Tree" << endl;
      //m_pointTree = Nabo::NNSearchF::createKDTreeLinearHeap( m_points ); 
      m_pointTree = Nabo::NNSearchF::createKDTreeLinearHeap( M ); 
   }


template<typename VertexT, typename NormalT>
   SearchTreeNabo< VertexT, NormalT >::~SearchTreeNabo() { 
    if( m_pointTree )
        delete m_pointTree;
   }


template<typename VertexT, typename NormalT>
   void SearchTreeNabo< VertexT, NormalT >::kSearch( coord< float > &qp, int neighbours, vector< ulong > &indices, vector< double > &distances )
   {
      Eigen::Vector3f q;
      q[0] = qp.x;
      q[1] = qp.y;
      q[2] = qp.z;

      Eigen::VectorXi ind( neighbours );
      Eigen::VectorXf dist( neighbours );
      
      m_pointTree->knn( q, ind, dist, neighbours );

      // store the found data
      indices.resize( ind.cols() );
      distances.resize( dist.cols() );
      for( size_t i(0); i < ind.cols(); i++ )
      {
        indices[i] = ind.cols();
      }
      for( size_t i(0); i < dist.cols(); i++ )
      {
        distances[i] = ind.cols();
      }
   }


/*
   Begin of radiusSearch implementations
*/
template<typename VertexT, typename NormalT>
   void SearchTreeNabo< VertexT, NormalT >::radiusSearch( float qp[3], double r, vector< ulong > &indices )
   {
      // clear possibly old information
      indices.clear();
      // keep track of found distances and indices
      vector< double > distances;

      double squared_radius = r*r;
      double max_radius = numeric_limits< double >::min();
      int k = 10;
      while( max_radius < r ){
         SearchTree< VertexT, NormalT >::kSearch( qp, k, indices, distances );

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


template<typename VertexT, typename NormalT>
   void SearchTreeNabo< VertexT, NormalT >::radiusSearch( VertexT& qp, double r, vector< ulong > &indices )
   {
      float qp_arr[3];
      qp_arr[0] = qp[0];
      qp_arr[1] = qp[1];
      qp_arr[2] = qp[2];
      this->radiusSearch( qp_arr, r, indices );
   }


template<typename VertexT, typename NormalT>
   void SearchTreeNabo< VertexT, NormalT >::radiusSearch( const VertexT& qp, double r, vector< ulong > &indices )
   {
      float qp_arr[3];
      qp_arr[0] = qp[0];
      qp_arr[1] = qp[1];
      qp_arr[2] = qp[2];
      this->radiusSearch( qp_arr, r, indices );
   }


template<typename VertexT, typename NormalT>
   void SearchTreeNabo< VertexT, NormalT >::radiusSearch( coord< float >& qp, double r, vector< ulong > &indices )
   {
      float qp_arr[3];
      qp_arr[0] = qp[0];
      qp_arr[1] = qp[1];
      qp_arr[2] = qp[2];
      this->radiusSearch( qp_arr, r, indices );
   }


template<typename VertexT, typename NormalT>
   void SearchTreeNabo< VertexT, NormalT >::radiusSearch( const coord< float >& qp, double r, vector< ulong > &indices )
   {
      float qp_arr[3];
      coord< float > qpcpy = qp;
      qp_arr[0] = qpcpy[0];
      qp_arr[1] = qpcpy[1];
      qp_arr[2] = qpcpy[2];
      this->radiusSearch( qp_arr, r, indices );
   }


} // namespace lssr
