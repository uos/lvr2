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
 * Normal.tcc
 *
 *  Created on: 10.02.2011
 *      Author: Thomas Wiemann
 */

namespace lssr
{

template<typename CoordType>
void Normal<CoordType>::normalize()
{
    //Don't normalize if we don't have to
    float l_square =
            this->x * this->x
            + this->y * this->y
            + this->z * this->z;

    if( fabs(1 - l_square) > 0.001){

        float length = sqrt(l_square);
        if(length != 0){
            this->x /= length;
            this->y /= length;
            this->z /= length;
        }
    }
}

} // namespace lssr
