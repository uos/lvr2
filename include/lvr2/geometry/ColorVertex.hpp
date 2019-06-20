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
 * ColorVertex.hpp
 *
 *  @date 17.06.2011
 *  @author Thomas Wiemann
 */

#ifndef COLORVERTEX_H_
#define COLORVERTEX_H_

#include "lvr2/geometry/BaseVector.hpp"

#include <ostream>

namespace lvr2
{


/**
 * @brief	A color vertex
 */
template<typename T, typename ColorT>
class ColorVertex : public BaseVector<T>
{
public:

    using CoordType = T;
	/**
	 * @brief	Default constructor. All coordinates and the color are initialized
	 * 			with zeros.
	 */
	ColorVertex()
	{
		this->x = this->y = this->z = 0;
		this->r = this->g = this->b = 0;
		fusion = false;
	}

	/**
	 * @brief	Builds a ColorVertex with the given coordinates.
	 */
	ColorVertex(const CoordType &_x, const CoordType &_y, const CoordType &_z)
	{
		this->x = _x;
		this->y = _y;
		this->z = _z;
		this->r = 0;
		this->g = 100;
		this->b = 0;
		fusion = false;
	}

	/**
	 * @brief	Builds a Vertex with the given coordinates.
	 */
	ColorVertex(const CoordType &_x, const CoordType &_y, const CoordType &_z,
			const unsigned char _r, const unsigned char _g, const unsigned char _b, ...)
	{
		this->x = _x;
		this->y = _y;
		this->z = _z;
		this->r = _r;
		this->g = _g;
		this->b = _b;
		fusion = false;
	}

	/**
	 * @brief	Copy Ctor.
	 */
	ColorVertex(const ColorVertex &o)
	{
		this->x = o.x;
		this->y = o.y;
		this->z = o.z;
		this->r = o.r;
		this->g = o.g;
		this->b = o.b;
		this->fusion = o.fusion;
	}

	/**
	 * @brief	Copy Ctor.
	 */
	ColorVertex(const T &o)
	{
		this->x = o.x;
		this->y = o.y;
		this->z = o.z;
		this->r = 0;
		this->g = 0;
		this->b = 0;
	}


    CoordType operator[](const int &index) const
    {

        switch ( index )
        {
            case 0: return this->x;
            case 1: return this->y;
            case 2: return this->z;
            case 3: return *((CoordType*) &r);
            case 4: return *((CoordType*) &g);
            case 5: return *((CoordType*) &b);
            case 6: return *((CoordType*) &fusion);
            default:
                throw std::overflow_error( "Access index out of range." );
        }
    }


    CoordType& operator[](const int &index)
    {
        switch ( index )
        {
            case 0: return this->x;
            case 1: return this->y;
            case 2: return this->z;
            case 3: return *((CoordType*) &r);
            case 4: return *((CoordType*) &g);
            case 5: return *((CoordType*) &b);
            case 6: return *((CoordType*) &fusion);
            default:
                throw std::overflow_error("Access index out of range.");
        }
    }


	ColorT r, g, b;
	bool fusion;

};

using uColorVertex = ColorVertex<float, unsigned char>;


/**
 * @brief	Output operator for color vertex types
 */
template<typename CoordType, typename ColorT>
inline std::ostream& operator<<(std::ostream& os, const ColorVertex<CoordType, ColorT> v){
	os << "ColorVertex: " << v.x << " " << v.y << " " << v.z << " " << (int)v.r << " " << (int)v.g << " " << (int)v.b << std::endl;
	return os;
}

} // namespace lvr22

#endif /* COLORVERTEX_H_ */
