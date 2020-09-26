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
 * OctreeTables.hpp
 *
 *  Created on: 17.01.2019
 *      Author: Benedikt Schumacher
 */

#ifndef OctreeTables_HPP_
#define OctreeTables_HPP_

namespace lvr2
{
const static int octreeNeighborTable[12][3] = {
  {12, 10,  9}, // 0
  {22, 12, 21}, // 1
  {16, 12, 15}, // 2
  { 4,  3, 12}, // 3
  {14, 10, 11}, // 4
  {23, 22, 14}, // 5
  {14, 16, 17}, // 6
  { 4,  5, 14}, // 7
  { 4,  1, 10}, // 8
  {22, 19, 10}, // 9
  { 4,  7, 16}, // 10
  {22, 25, 16}  // 11
};

const static int octreeNeighborVertexTable[12][3] = {
  { 4,  2,  6},
  { 3,  5,  7},
  { 0,  6,  4},
  { 1,  5,  7},
  { 0,  6,  2},
  { 3,  7,  1},
  { 2,  4,  0},
  { 5,  1,  3},
  { 9, 11, 10},
  { 8, 10, 11},
  {11,  9,  8},
  {10,  8,  9}
};

const static int octreeVertexTable[8][3] = {
  {-1, -1, -1},
  { 1, -1, -1},
  { 1,  1, -1},
  {-1,  1, -1},
  {-1, -1,  1},
  { 1, -1,  1},
  { 1,  1,  1},
  {-1,  1,  1}
};

const static int octreeCenterTable[8][3] = {
  {-1, -1, -1},
  { 1, -1, -1},
  {-1,  1, -1},
  { 1,  1, -1},
  {-1, -1,  1},
  { 1, -1,  1},
  {-1,  1,  1},
  { 1,  1,  1}
};

const static int octreeCornerNeighborTable[64][3] = {
  // 0
  {-1, -1, -1},
  { 0, -1, -1},
  {-1,  0, -1},
  { 0,  0, -1},
  {-1, -1,  0},
  { 0, -1,  0},
  {-1,  0,  0},
  { 0,  0,  0},
  // 1
  { 0, -1, -1},
  { 1, -1, -1},
  { 0,  0, -1},
  { 1,  0, -1},
  { 0, -1,  0},
  { 1, -1,  0},
  { 0,  0,  0},
  { 1,  0,  0},
  // 2
  {-1,  0, -1},
  { 0,  0, -1},
  {-1,  1, -1},
  { 0,  1, -1},
  {-1,  0,  0},
  { 0,  0,  0},
  {-1,  1,  0},
  { 0,  1,  0},
  // 3
  { 0,  0, -1},
  { 1,  0, -1},
  { 0,  1, -1},
  { 1,  1, -1},
  { 0,  0,  0},
  { 1,  0,  0},
  { 0,  1,  0},
  { 1,  1,  0},
  // 4
  {-1, -1,  0},
  { 0, -1,  0},
  {-1,  0,  0},
  { 0,  0,  0},
  {-1, -1,  1},
  { 0, -1,  1},
  {-1,  0,  1},
  { 0,  0,  1},
  // 5
  { 0, -1,  0},
  { 1, -1,  0},
  { 0,  0,  0},
  { 1,  0,  0},
  { 0, -1,  1},
  { 1, -1,  1},
  { 0,  0,  1},
  { 1,  0,  1},
  // 6
  {-1,  0,  0},
  { 0,  0,  0},
  {-1,  1,  0},
  { 0,  1,  0},
  {-1,  0,  1},
  { 0,  0,  1},
  {-1,  1,  1},
  { 0,  1,  1},
  // 7
  { 0,  0,  0},
  { 1,  0,  0},
  { 0,  1,  0},
  { 1,  1,  0},
  { 0,  0,  1},
  { 1,  0,  1},
  { 0,  1,  1},
  { 1,  1,  1}
};

const static int edgeDistanceTable[12][2] = {
  {0, 1},
  {0, 2},
  {0, 4},
  {1, 3},
  {1, 3},
  {2, 3},
  {2, 6},
  {3, 7},
  {4, 5},
  {4, 6},
  {5, 7},
  {6, 7}
};

} // namespace lvr2

#endif /* OctreeTables_HPP_ */
