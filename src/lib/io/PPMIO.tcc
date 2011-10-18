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
 * PPMIO.cpp
 *
 *  Created on: 08.09.2011
 *      Author: pg2011
 */

namespace lssr
{

template<typename ColorType>
PPMIO<ColorType>::PPMIO()
{
    m_data    = 0;
    m_sizeX   = 0;
    m_sizeY   = 0;
}

template<typename ColorType>
void PPMIO<ColorType>::setDataArray(ColorType** array, size_t sizeX, size_t sizeY)
{
    m_data = array;
    m_sizeX = sizeX;
    m_sizeY = sizeY;
}


template<typename ColorType>
void PPMIO<ColorType>::write(string filename)
{
    ofstream out(filename.c_str());

    if(out.good())
    {
    	out<<"P6"<<" "<<m_sizeX<<" "<<m_sizeY<<" "<<"255"<<endl;
    	for(size_t y = 0; y<m_sizeY; y++)
    		for(size_t x = 0; x<m_sizeX; x++)
    			out<<m_data[y][x].r<<m_data[y][x].g<<m_data[y][x].b;
    }

    out.close();

}

}
