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
*  SearchTree.hpp
*
*       Created on: 02.01.2012
*           Author: Florian Otte
 */

#ifndef SEARCH_TREE_H_
#define SEARCH_TREE_H_

/*#include "../io/Model.hpp"
#include "../io/Progress.hpp"
#include "../io/Timestamp.hpp"
#include "../io/PLYIO.hpp"
#include "../io/AsciiIO.hpp"
#include "../io/UosIO.hpp" */


// Standard C++ includes
#include <vector>
#include <iostream>

using std::cout;
using std::endl;

typedef unsigned long ulong;

namespace lssr
{

/**
 * @brief Abstract interface for storing and
 *        searching through a set of points.
 *        Query functions for nearest neighbour searches
 *        are defined.
 */

template< typename VertexT>
class SearchTree
{
public:

    typedef boost::shared_ptr< SearchTree< VertexT> > Ptr;

    /**
     * @brief This function performs a k-next-neightbour search on the
                     data that were given in the conrstructor.

     * @param qp          A float array which contains the query point for which the neighbours are searched.
     * @param k           The number of neighbours that should be searched.
     * @param indices     A vector that stores the indices for the neighbours whithin the dataset.
     */
    virtual void kSearch( float              qp[3], int k, vector< ulong > &indices );
    virtual void kSearch( VertexT&              qp, int k, vector< ulong > &indices );
    virtual void kSearch( const VertexT&        qp, int k, vector< ulong > &indices );
    virtual void kSearch( const coord< float >& qp, int k, vector< ulong > &indices );
    virtual void kSearch( coord< float >&       qp, int k, vector< ulong > &indices );

    /**
     * @brief This function performs a k-next-neightbour search on the
                     data that were given in the conrstructor.

     * @param qp          A float array which contains the query point for which the neighbours are searched.
     * @param neighbours  The number of neighbours that should be searched.
     * @param indices     A vector that stores the indices for the neighbours whithin the dataset.
     * @param distances   A vector that sotres the distances for the neighbours that are found.
     */
    virtual void kSearch( float               qp[3], int k, vector< ulong > &indices, vector< double > &distances );
    virtual void kSearch( VertexT&               qp, int k, vector< ulong > &indices, vector< double > &distances );
    virtual void kSearch( const VertexT&         qp, int k, vector< ulong > &indices, vector< double > &distances );
    virtual void kSearch( const coord < float >& qp, int k, vector< ulong > &indices, vector< double > &distances );

    // Pure virtual. All other search functions map to this. Must be implemented in sub-class.
    virtual void kSearch( coord < float >&       qp, int k, vector< ulong > &indices, vector< double > &distances ) = 0;
    virtual void kSearch( VertexT      qp, int k, vector< VertexT > &neighbors ) = 0;



    virtual void radiusSearch( float              qp[3], double r, vector< ulong > &indices ) = 0;
    virtual void radiusSearch( VertexT&              qp, double r, vector< ulong > &indices ) = 0;
    virtual void radiusSearch( const VertexT&        qp, double r, vector< ulong > &indices ) = 0;
    virtual void radiusSearch( coord< float >&       qp, double r, vector< ulong > &indices ) = 0;
    virtual void radiusSearch( const coord< float >& qp, double r, vector< ulong > &indices ) = 0;


    /**
     * @brief Set the number of neighbours used to estimate and interpolate normals.
     */
    virtual void setKn( int kn );


    /**
     * @brief Set the number of neighbours used to estimate and interpolate normals.
     */
    virtual void setKi( int ki );


    /**
     * @brief Set the number of neighbours used for normal estimation
     */
    virtual void setKd( int kd );


    /**
     * @brief Get the number of neighbours used for normal interpolation
     */
    virtual int getKn( void );


    /**
     * @brief Get the number of tangent planes used for distance determination
     */
    virtual int getKi( void );


    /**
     * @brief Get the number of neighbours used to estimate and interpolate normals.
     */
    int getKd( void );


protected:

    /// The number of neighbors used for initial normal estimation
    int                         m_kn;

    /// The number of neighbors used for normal interpolation
    int                         m_ki;

    /// The number of tangent planes used for distance determination
    int                         m_kd;

}; // SearchTreeClass.

// include implementation for this class

}  // namespace lssr{;
#include "SearchTree.tcc"
#endif // include-guard
