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
 * ColorVertex.hpp
 *
 *  @date 17.06.2011
 *  @author Thomas Wiemann
 */

#ifndef COLORVERTEX_H_
#define COLORVERTEX_H_

#include "Vertex.hpp"

namespace lssr
{

/**
 * @brief	A color vertex
 */
template<typename CoordType, typename ColorT>
class ColorVertex : public Vertex<CoordType>
{
public:

	/**
	 * @brief	Default constructor. All coordinates and the color are initialized
	 * 			with zeros.
	 */
	ColorVertex()
	{
		this->x = this->y = this->z = 0;
		this->r = this->g = this->b = 0;
	}

	/**
	 * @brief	Builds a ColorVertex with the given coordinates.
	 */
	ColorVertex(const CoordType &_x, const CoordType &_y, const CoordType &_z)
	{
		this->x = _x;
		this->y = _y;
		this->z = _z;
		this->r = 255;
		this->g = 255;
		this->b = 0;
	}

	/**
	 * @brief	Builds a Vertex with the given coordinates.
	 */
	ColorVertex(const CoordType &_x, const CoordType &_y, const CoordType &_z,
			const unsigned char _r, const unsigned char _g, const unsigned char _b)
	{
		this->x = _x;
		this->y = _y;
		this->z = _z;
		this->r = _r;
		this->g = _g;
		this->b = _b;
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
	}

	/**
	 * @brief	Copy Ctor.
	 */
	ColorVertex(const Vertex<CoordType> &o)
	{
		this->x = o.x;
		this->y = o.y;
		this->z = o.z;
		this->r = 0;
		this->g = 0;
		this->b = 0;
	}

	ColorT r, g, b;

};

typedef ColorVertex<float, unsigned char> uColorVertex;


/**
 * @brief	Output operator for color vertex types
 */
template<typename CoordType, typename ColorT>
inline ostream& operator<<(ostream& os, const ColorVertex<CoordType, ColorT> v){
	os << "ColorVertex: " << v.x << " " << v.y << " " << v.z << " " << (int)v.r << " " << (int)v.g << " " << (int)v.b << endl;
	return os;
}


} // namespace lssr

#endif /* COLORVERTEX_H_ */
