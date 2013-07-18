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
*  SearchTreeStann.hpp
*
*       Created on: 02.01.2012
*           Author: Florian Otte
 */

#ifndef SEARCHTREESTANN_H_
#define SEARCHTREESTANN_H_

// C++ Stl includes
#include <vector>

// Stann
#include "../stann/sfcnn.hpp"

// Superclass
#include "SearchTree.hpp"

using std::vector;

namespace lvr {

/**
 * @brief SearchClass for point data.
 *
 *      This class uses the stann( https://sites.google.com/a/compgeom.com/stann/ )
 *      library to implement a nearest neighbour search for point-data.
 */
template< typename VertexT >
class SearchTreeStann : public SearchTree< VertexT >
{
public:

    typedef boost::shared_ptr< SearchTreeStann< VertexT > > Ptr;


    /**
     *  @brief Constructor. Takes the point-data and initializes the underlying searchtree.
     *
     *  @param loader  A PointBuffer point that holds the data.
     *  @param kn      The number of neighbour points used for normal estimation.
     *  @param ki      The number of neighbour points used for normal interpolation.
     *  @param kd      The number of neighbour points esed for distance value calculation.
     */
    SearchTreeStann( PointBufferPtr points,
            size_t &n_points,
            const int &kn = 10,
            const int &ki = 10,
            const int &kd = 10,
            const bool &useRansac = false );


    /**
     * @brief Destructor
     */
    virtual ~SearchTreeStann();

    /**
     * @brief This function performs a k-next-neightbour search on the
                     data that were given in the conrstructor.

     * @param qp          A float array which contains the query point for which the neighbours are searched.
     * @param neighbours  The number of neighbours that should be searched.
     * @param indices     A vector that stores the indices for the neighbours whithin the dataset.
     * @param distances   A vector that sotres the distances for the neighbours that are found.
     */
    virtual void kSearch( coord < float >& qp, int neighbours, vector< ulong > &indices, vector< double > &distances );
    virtual void kSearch(VertexT qp, int k, vector< VertexT > &neighbors);

    virtual void radiusSearch( float              qp[3], double r, vector< ulong > &indices );
    virtual void radiusSearch( VertexT&              qp, double r, vector< ulong > &indices );
    virtual void radiusSearch( const VertexT&        qp, double r, vector< ulong > &indices );
    virtual void radiusSearch( coord< float >&       qp, double r, vector< ulong > &indices );
    virtual void radiusSearch( const coord< float >& qp, double r, vector< ulong > &indices );

protected:

    /// Store wether to use a randomized algorithm for plane calculation
    bool    m_useRansac;

    /// Store the stann kd-tree
    sfcnn< coord< float >, 3, float >   m_pointTree;

    /// The points
    coord3fArr                          m_points;

    /// The colors
    color3bArr							m_colors;
}; // SearchTreeStann


} // namespace lvr
#include "SearchTreeStann.tcc"

#endif  // include-guard
