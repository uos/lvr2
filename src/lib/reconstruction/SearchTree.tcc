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
 *  SearchTree.tcc
 *
 *       Created on: 02.01.2012
 *           Author: Florian Otte
 */

namespace lssr {

template<typename VertexT>
void SearchTree< VertexT >::kSearch( float qp[3], int neighbours, vector< ulong > &indices )
{
    vector< double > distances;
    this->kSearch( qp, neighbours, indices, distances);
}


template<typename VertexT>
void SearchTree< VertexT >::kSearch( VertexT &qp, int neighbours, vector< ulong > &indices )
{
    vector< double > distances;
    this->kSearch( qp, neighbours, indices, distances);
}


template<typename VertexT>
void SearchTree< VertexT >::kSearch( const VertexT &qp, int neighbours, vector< ulong > &indices )
{
    vector< double > distances;
    this->kSearch( qp, neighbours, indices, distances);
}


template<typename VertexT>
void SearchTree< VertexT >::kSearch( coord< float > &qp, int neighbours, vector< ulong > &indices )
{
    vector< double > distances;
    this->kSearch( qp, neighbours, indices, distances);
}


template<typename VertexT>
void SearchTree< VertexT >::kSearch( const coord< float > &qp, int neighbours, vector< ulong > &indices )
{
    vector< double > distances;
    this->kSearch( qp, neighbours, indices, distances);
}


/*
   Begin of kSearch implementations with distances
 */
template<typename VertexT>
void SearchTree< VertexT >::kSearch( float qp[3], int neighbours, vector< ulong > &indices, vector< double > &distances )
{
    coord< float > Point;
    Point[0] = qp[0];
    Point[1] = qp[1];
    Point[2] = qp[2];
    this->kSearch( Point, neighbours, indices, distances );
}


template<typename VertexT>
void SearchTree< VertexT >::kSearch( VertexT &qp, int neighbours, vector< ulong > &indices, vector< double > &distances )
{
    float qp_arr[3];
    qp_arr[0] = qp[0];
    qp_arr[1] = qp[1];
    qp_arr[2] = qp[2];
    this->kSearch( qp_arr, neighbours, indices, distances );
}


template<typename VertexT>
void SearchTree< VertexT >::kSearch( const VertexT &qp, int neighbours, vector< ulong > &indices, vector< double > &distances )
{
    float qp_arr[3];
    qp_arr[0] = qp[0];
    qp_arr[1] = qp[1];
    qp_arr[2] = qp[2];
    this->kSearch( qp_arr, neighbours, indices, distances);
}


template<typename VertexT>
void SearchTree< VertexT >::kSearch( const coord< float > &qp, int neighbours, vector< ulong > &indices, vector< double > &distances )
{
    float qp_arr[3];
    coord< float > qpcpy = qp;
    qp_arr[0] = qpcpy[0];
    qp_arr[1] = qpcpy[1];
    qp_arr[2] = qpcpy[2];
    this->kSearch( qp, neighbours, indices, distances);
}


template<typename VertexT>
void SearchTree< VertexT >::setKn( int kn ) {
    m_kn = kn;
}


template<typename VertexT>
void SearchTree< VertexT >::setKi( int ki ) {
    m_ki = ki;
}


template<typename VertexT>
void SearchTree< VertexT >::setKd( int kd ) {
    m_kd = kd;
}

template<typename VertexT>
int SearchTree< VertexT >::getKn() {
    return m_kn;
}


template<typename VertexT>
int SearchTree< VertexT >::getKi() {
    return m_ki;
}

template<typename VertexT>
int SearchTree< VertexT >::getKd() {
    return m_kd;
}
} // namespace
