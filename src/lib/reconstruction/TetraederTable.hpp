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
 * TetraederTable.hpp
 *
 *  @date 23.11.2011
 *  @author Thomas Wiemann
 */

#ifndef TETRAEDERTABLE_HPP_
#define TETRAEDERTABLE_HPP_

namespace lssr
{

const static int TetraederTable[16][7] = {
        {-1, -1, -1, -1, -1, -1, -1},   //0
        { 0,  2,  1, -1, -1, -1, -1},   //1
        { 0,  3,  5, -1, -1, -1, -1},   //2
        { 5,  2,  1,  5,  3,  1, -1},   //3 ??
        { 1,  4,  3, -1, -1, -1, -1},   //4
        { 0,  3,  2,  3,  4,  2, -1},   //5
        { 5,  1,  0,  5,  4,  1, -1},   //6 ??
        { 5,  4,  2, -1, -1, -1, -1},   //7
        { 5,  2,  4, -1, -1, -1, -1},   //8
        { 5,  0,  1,  5,  1,  4, -1},   //9 ??
        { 0,  2,  3,  3,  2,  4, -1},   //10
        { 4,  1,  3, -1, -1, -1, -1},   //11
        { 5,  1,  2,  5,  1,  3, -1},   //12 ??
        { 0,  5,  3, -1, -1, -1, -1},   //13
        { 0,  1,  2, -1, -1, -1, -1},   //14
        {-1, -1, -1, -1, -1, -1, -1}    //16
};

} /* namespace lssr */
#endif /* TETRAEDERTABLE_HPP_ */
