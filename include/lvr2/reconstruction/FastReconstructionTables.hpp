/**
 * Copyright (c) 2018, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

 /*
 * MCReconstructionTables.hpp
 *
 *  Created on: 02.03.2011
 *      Author: Thomas Wiemann
 */

#ifndef MCRECONSTRUCTIONTABLES_HPP_
#define MCRECONSTRUCTIONTABLES_HPP_

namespace lvr2
{

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

const static int box_neighbour_table[8][3] = {
    { 1, 3, 4},
    { 0, 2, 5},
    { 3, 1, 6},
    { 2, 0, 7},
    { 5, 7, 0},
    { 4, 6, 1},
    { 7, 5, 2},
    { 6, 4, 3}
};

/**
 * @brief This tables stors adjacency information for the
 *        grid creation algorithm.
 */

const static int TSDFCreateTable[8][3] = {
  { 0,  0,  0},
  { 1,  0,  0}, 
  { 1,  1,  0}, 
  { 0,  1,  0},
  { 0,  0,  1}, 
  { 1,  0,  1}, 
  { 1,  1,  1},
  { 0,  1,  1}
};

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

} // namespace lvr2

#endif /* MCRECONSTRUCTIONTABLES_HPP_ */
