/*
 * FastBoxTables.hpp
 *
 *  Created on: 09.03.2011
 *      Author: Thomas Wiemann
 */

#ifndef FASTBOXTABLES_HPP_
#define FASTBOXTABLES_HPP_

namespace lssr
{

const static int neighbor_table[12][3] = {
  {12, 10,  9},
  {22, 12, 21},
  {16, 12, 15},
  { 4,  3, 12},
  {14, 10, 11},
  {23, 22, 14},
  {14, 16, 17},
  { 4,  5, 14},
  { 4,  1, 10},
  {22, 19, 10},
  { 4,  7, 16},
  {22, 25, 16}
};

const static int neighbor_vertex_table[12][3] = {
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

} // namespace lssr
#endif /* FASTBOXTABLES_HPP_ */
