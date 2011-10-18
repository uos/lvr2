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
 * MCReconstructionTables.hpp
 *
 *  Created on: 02.03.2011
 *      Author: Thomas Wiemann
 */

#ifndef MCRECONSTRUCTIONTABLES_HPP_
#define MCRECONSTRUCTIONTABLES_HPP_

/**
 * @brief A table coding the relations between shared vertices of
 *        adjacent positions in the grid created during the marching
 *        cubes reconstruction process
 *
 * Each box corner in the grid is shared with 7 other boxes.
 * To find an already existing corner, these boxes have to
 * be checked. The following table holds the information where
 * to look for a given corner. The coding is as follows:
 *
 * Table row = query vertex
 *
 * Each row consists of 7 quadruples. The first three numbers
 * indicate, how the indices in x- y- and z-direction have to
 * be modified. The fourth entry is the vertex of the box
 * correspondig to the modified indices.
 *
 * <i>Example</i>: index_x = 10, index_y = 7, index_z = 5
 *
 * <i>Query vertex</i> = 5
 *
 * First quadruple: {+1, 0, +1, 0} Indices pointing to the nb-box:
 * 10 + 1, 7 + 0, 5 + 1.
 * --> The first shared vertex is vertex number 0 of the box in position
 * (11, 7, 6) of the grid.
 *
 * Simple isn't it?
 */
const static int shared_vertex_table[8][28] = {
    {-1, 0, 0, 1, -1, -1, 0, 2,  0, -1, 0, 3, -1,  0, -1, 5, -1, -1, -1, 6,  0, -1, -1, 7,  0,  0, -1, 4},
    { 1, 0, 0, 0,  1, -1, 0, 3,  0, -1, 0, 2,  0,  0, -1, 5,  1,  0, -1, 4,  1, -1, -1, 7,  0, -1, -1, 6},
    { 1, 1, 0, 0,  0,  1, 0, 1,  1,  0, 0, 3,  1,  1, -1, 4,  0,  1, -1, 5,  0,  0, -1, 6,  1,  0, -1, 7},
    { 0, 1, 0, 0, -1,  1, 0, 1, -1,  0, 0, 2,  0,  1, -1, 4, -1,  1, -1, 5, -1,  0, -1, 6,  0,  0, -1, 7},
    { 0, 0, 1, 0, -1,  0, 1, 1, -1, -1, 1, 2,  0, -1,  1, 3, -1,  0,  0, 5, -1, -1,  0, 6,  0, -1,  0, 7},
    { 1, 0, 1, 0,  0,  0, 1, 1,  0, -1, 1, 2,  1, -1,  1, 3,  1,  0,  0, 4,  0, -1,  0, 6,  1, -1,  0, 7},
    { 1, 1, 1, 0,  0,  1, 1, 1,  0,  0, 1, 2,  1,  0,  1, 3,  1,  1,  0, 4,  0,  1,  0, 5,  1,  0,  0, 7},
    { 0, 1, 1, 0, -1,  1, 1, 1, -1,  0, 1, 2,  0,  0,  1, 3,  0,  1,  0, 4, -1,  1,  0, 5, -1,  0,  0, 6}
};


/**
 * @brief This table states where each coordinate of a box vertex
 *        is relative to the box center
 */
const static int box_creation_table[8][3] = {
    {-1, -1, -1},
    { 1, -1, -1},
    { 1,  1, -1},
    {-1,  1, -1},
    {-1, -1,  1},
    { 1, -1,  1},
    { 1,  1,  1},
    {-1,  1,  1}
};

/**
 * @brief This tables stors adjacency information for the
 *        grid creation algorithm.
 */
const static int HGCreateTable[8][3] = {
  { 0,  0,  0}, 
  {-1,  0,  0}, 
  {-1,  0, -1}, 
  { 0,  0, -1},
  { 0, -1,  0}, 
  {-1, -1,  0}, 
  {-1, -1, -1},
  { 0, -1, -1}
};

#endif /* MCRECONSTRUCTIONTABLES_HPP_ */
