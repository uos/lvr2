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

/**
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

#pragma once


#include <boost/shared_array.hpp>

#include "lvr2/display/GlTexture.hpp"

#include <map>
#include <vector>

namespace lvr2
{


template<typename CoordT>
struct coord
{
	CoordT x;
	CoordT y;
	CoordT z;
	CoordT& operator[]( const size_t i )
	{
		switch ( i )
		{
		case 0:
			return x;
			break;
		case 1:
			return y;
			break;
		case 2:
			return z;
			break;
		default:
			return z;
		}
	}
};


template<typename ColorT>
struct color
{
	ColorT r;
	ColorT g;
	ColorT b;
	ColorT& operator[] ( const size_t i )
	{
		switch ( i )
		{
		case 0:
			return r;
			break;
		case 1:
			return g;
			break;
		case 2:
			return b;
			break;
		default:
			return b;
		}
	}
};


template<typename T>
struct idxVal
{
	T value;
	T& operator[] ( const size_t i )
	{
		return value;
	}
};



struct RGBMaterial
{
	unsigned char r;
	unsigned char g;
	unsigned char b;
	int texture_index;
};

typedef boost::shared_array<int> intArray;

typedef boost::shared_array<unsigned int> indexArray;

typedef boost::shared_array<unsigned int> uintArr;


typedef boost::shared_array<float> floatArr;
typedef boost::shared_array<double> doubleArr;


typedef boost::shared_array<unsigned char> ucharArr;

typedef boost::shared_array<short> shortArr;

typedef boost::shared_array< color<unsigned char> > color3bArr;


typedef boost::shared_array< coord<float> > coord3fArr;


typedef boost::shared_array< idxVal<float> > idx1fArr;


typedef boost::shared_array< coord<unsigned int> > idx3uArr;


typedef boost::shared_array< idxVal<unsigned int> > idx1uArr;


typedef boost::shared_array< RGBMaterial* > materialArr;


typedef boost::shared_array< GlTexture* > textureArr;


typedef std::pair<size_t, size_t> indexPair;


typedef std::map<std::string, std::vector<unsigned int> > labeledFacesMap;
}

