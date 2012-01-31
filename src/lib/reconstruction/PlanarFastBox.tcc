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
 * PlanarFastBox.cpp
 *
 *  @date 27.01.2012
 *  @author Thomas Wiemann
 */

namespace lssr
{

static int MSQTable_51[16][7] =
{
        {-1, -1, -1, -1, -1, -1, -1},  // 0
        { 3, 11, 12, -1, -1, -1, -1},  // 1
        {14,  1, 16, -1, -1, -1, -1},  // 2
        { 3,  1, 12,  1, 13, 12, -1},  // 3
        {17, 16,  5, -1, -1, -1, -1},  // 4
        {14,  1, 16, -1, -1, -1, -1},  // 5
        { 3, 14, 17,  3, 17,  7, -1},  // 6
        {12, 17,  7, -1, -1, -1, -1},  // 7
        {12, 17,  7, -1, -1, -1, -1},  // 8
        {14,  1,  5, -1, -1, -1, -1},  // 9
        { 3, 14, 12, -1, -1, -1, -1},  // 10
        {17, 16,  5, -1, -1, -1, -1},  // 11
        {12, 16,  5, 12,  5,  7, -1},  // 12
        {14,  1, 16, -1, -1, -1, -1},  // 13
        { 3, 14, 12, -1, -1, -1, -1},  // 14
        {-1, -1, -1, -1, -1, -1, -1},  // 15
};

template<typename VertexT, typename NormalT>
PlanarFastBox<VertexT, NormalT>::PlanarFastBox(VertexT &center)
    : TetraederBox<VertexT, NormalT>(center)
{

}

template<typename VertexT, typename NormalT>
void PlanarFastBox<VertexT, NormalT>::getSurface(
        BaseMesh<VertexT, NormalT> &mesh,
        vector<QueryPoint<VertexT> > &query_points,
        uint &globalIndex)
{

}


} /* namespace lssr */
