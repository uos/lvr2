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
 *           Author: Florian Otte, Thomas Wiemann
 *
 */

#include <lvr/io/Timestamp.hpp>

#include <iostream>
using std::cout;
using std::endl;

namespace lvr2 {

// template<typename BaseVecT>
// void SearchTree<BaseVecT>::initBuffers(PointBufferPtr buffer)
// {
//     this->m_pointData = buffer->getPointArray(this->m_numPoints);
//     this->m_haveColors = false;
// }

template<typename BaseVecT>
void SearchTree<BaseVecT>::kSearch(
    const Point<BaseVecT> &qp,
    int neighbours,
    vector<int>& indices
)
{
    vector<float> distances;
    this->kSearch(qp, neighbours, indices, distances);
}



template<typename BaseVecT>
void SearchTree<BaseVecT>::setKn(int kn)
{
    m_kn = kn;
}


template<typename BaseVecT>
void SearchTree<BaseVecT>::setKi(int ki)
{
    m_ki = ki;
}


template<typename BaseVecT>
void SearchTree<BaseVecT>::setKd(int kd)
{
    m_kd = kd;
}

template<typename BaseVecT>
int SearchTree<BaseVecT>::getKn()
{
    return m_kn;
}


template<typename BaseVecT>
int SearchTree<BaseVecT>::getKi()
{
    return m_ki;
}

template<typename BaseVecT>
int SearchTree<BaseVecT>::getKd()
{
    return m_kd;
}

} // namespace
