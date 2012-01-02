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
 * SearchTreeStann.tcc 
 *
 *  Created on: 02.01.2012
 *      Author: Florian Otte
 */

// stl includes
#include <limit>

// External libraries in lssr source tree
#include "../Eigen/Dense"

// boost libraries
#include <boost/filesystem.hpp>

// lssr includes

using std::cout;
using std::endl;
using std::numeric_limits< double >;

namespace lssr {

template<typename VertexT, typename NormalT>
   SearchTreeStann< VertexT, NormalT >::SearchTreeStann( coord3fArr *points, int &n_points, const int &kn, const int &ki, const int &kd, const bool &useRansac ) 
   {

      // Make sure that point information is given
      assert(*points);

      // Store parameters
      this->m_ki = ki;
      this->m_kn = kn;
      this->m_kd = kd;
      m_useRansac = useRansac;

      // Create Stann Kd-tree
      cout << timestamp << "Creating STANN Kd-Tree" << endl;
      m_pointTree = sfcnn< coord< float >, 3, float >( points->get(), n_points, omp_get_num_procs() );
   }


template<typename VertexT, typename NormalT>
   void SearchTreeStann< VertexT, NormalT >::kSearch( float[3] qp, int neighbours, vector< ulong > &indices )
   {
      vector< double > distances;
      this->kSearch( qp, neighbours, indices, distances, 0);
   }


template<typename VertexT, typename NormalT>
   void SearchTreeStann< VertexT, NormalT >::kSearch( VertexT &qp, int neighbours, vector< ulong > &indices )
   {
      vector< double > distances;
      this->kSearch( qp, neighbours, indices, distances, 0);
   }


template<typename VertexT, typename NormalT>
   void SearchTreeStann< VertexT, NormalT >::kSearch( const VertexT &qp, int neighbours, vector< ulong > &indices )
   {
      vector< double > distances;
      this->kSearch( qp, neighbours, indices, distances, 0);
   }

template<typename VertexT, typename NormalT>
   void SearchTreeStann< VertexT, NormalT >::kSearch( float[3] qp, int neighbours, vector< ulong > &indices, vector< float > &distances )
   {
      m_pointTree.ksearch( qp, neighbours, indices, distances, 0);
   }


template<typename VertexT, typename NormalT>
   void SearchTreeStann< VertexT, NormalT >::kSearch( VertexT &qp, int neighbours, vector< ulong > &indices, vector< float > &distances )
   {
      float qp_arr[3];
      qp_arr[0] = qp[0];
      qp_arr[1] = qp[1];
      qp_arr[2] = qp[2];
      m_pointTree.ksearch( qp_arr, neighbours, indices, distances, 0);
   }


template<typename VertexT, typename NormalT>
   void SearchTreeStann< VertexT, NormalT >::kSearch( const VertexT &qp, int neighbours, vector< ulong > &indices, vector< float > &distances )
   {
      vector< double > distances;
      float qp_arr[3];
      qp_arr[0] = qp[0];
      qp_arr[1] = qp[1];
      qp_arr[2] = qp[2];
      m_pointTree.ksearch( qp_arr, neighbours, indices, distances, 0);
   }

template<typename VertexT, typename NormalT>
   void SearchTreeStann< VertexT, NormalT >::radiusSearch( float[3] qp, double r, vector< ulong > &indices )
   {
      // clear possibly old information
      indices.clear();
      // keep track of found distances and indices
      vector< float > distances;

      double squared_radius = r*r;
      double max_radius = numeric_limits< double >::min();
      int k = 10;
      while( max_radius < r ){
         this->kSearch( qp, k, resV, distances );

         // check distances for all neighbours
         for( int i=0; i < distances.length(); i++ )
         {
            max_radius = max( max_radius, distances[i] );
            if( distances[i] < r )
            {
               indices.push_back( indices[i] );
            }
         }
         k *= 2;
      }
   }

template<typename VertexT, typename NormalT>
   void SearchTreeStann< VertexT, NormalT >::radiusSearch( VertexT& qp, double r, vector< ulong > &indices )
   {
      double qp_arr[3];
      qp_arr[0] = qp[0];
      qp_arr[1] = qp[1];
      qp_arr[2] = qp[2];
      this->radiusSearch( qp_arr, r, resV, resN );
   }

template<typename VertexT, typename NormalT>
   void SearchTreeStann< VertexT, NormalT >::radiusSearch( const VertexT& qp, double r, vector< ulong > &indices )
   {
      double qp_arr[3];
      qp_arr[0] = qp[0];
      qp_arr[1] = qp[1];
      qp_arr[2] = qp[2];
      this->radiusSearch( qp_arr, r, resV, resN );
   }
} // namespace lssr
