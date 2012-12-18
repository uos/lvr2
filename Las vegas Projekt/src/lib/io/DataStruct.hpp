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
 *
 * @file      DataStruct.hpp
 * @brief     Datastructures for holding loaded data.
 * @details   
 * 
 * @author    Lars Kiesow (lkiesow), lkiesow@uos.de, Universität Osnabrück
 * @version   111129
 * @date      Recreated:     2011-11-29 12:33:48
 * @date      Last modified: 2011-11-29 12:33:51
 *
 **/

#ifndef DATASTRUCT_HPP_INCLUDED
#define DATASTRUCT_HPP_INCLUDED


#include "boost/shared_array.hpp"
#include "display/GlTexture.hpp"

namespace lssr
{

typedef unsigned char uchar;


struct Material
{
	uchar r;
	uchar g;
	uchar b;
	int texture_index;
};


template<typename CoordT>
struct coord
{
    CoordT x;
    CoordT y;
    CoordT z;
    CoordT& operator[]( const size_t i ) {
        switch ( i ) {
            case 0: return x; break;
            case 1: return y; break;
            case 2: return z; break;
            default: return z;
        }
    }
};


template<typename ColorT>
struct color
{
    ColorT r;
    ColorT g;
    ColorT b;
    ColorT& operator[] ( const size_t i ) {
        switch ( i ) {
            case 0: return r; break;
            case 1: return g; break;
            case 2: return b; break;
            default: return b;
        }
    }
};


template<typename T>
struct idxVal
{
    T value;
    T& operator[] ( const size_t i ) {
        return value;
    }
};


typedef boost::shared_array<float> floatArr;


typedef boost::shared_array<uchar> ucharArr;


typedef boost::shared_array<unsigned int> uintArr;


typedef boost::shared_array< color<uchar> > color3bArr;


typedef boost::shared_array< coord<float> > coord3fArr;


typedef boost::shared_array< idxVal<float> > idx1fArr;


typedef boost::shared_array< coord<unsigned int> > idx3uArr;


typedef boost::shared_array< idxVal<unsigned int> > idx1uArr;

typedef boost::shared_array< Material* > materialArr;

typedef boost::shared_array< GlTexture* > textureArr;

typedef std::pair<size_t, size_t> indexPair;

}

#endif
