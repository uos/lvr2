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
 * Normal.hpp
 *
 *  Created on: 04.02.2011
 *      Author: Thomas Wiemann
 */

#ifndef NORMAL_HPP_
#define NORMAL_HPP_


#include "./Vertex.hpp"

namespace lvr {

/**
 * @brief	A normal implementation. Basically a vertex
 * 			with normalization functionality.
 */
template<typename CoordType>
class Normal : public lvr::Vertex<CoordType>
{

public:

	/**
	 * @brief	Default constructor. All elements are set
	 * 			to zero.
	 */
	Normal() : Vertex<CoordType>() {};

	/**
	 * @brief	Constructs a new normal from the given
	 * 			components. Applies normalization.
	 */
	Normal(CoordType x, CoordType y, CoordType z)
	: Vertex<CoordType>(x, y, z)
	  {
		normalize();
	  }

	/**
	 * @brief	Copy constructor for vertices. Applies
	 * 			normalization.
	 */
	template<typename T>
	Normal(const Vertex<T> &other) : Vertex<T>(other)
	{
		normalize();
	}

	/**
	 * @brief	Constructs from another normal. Implemented
	 * 			to avoid unnecessary normalizations.
	 */
	template<typename T>
	Normal(const Normal &other)
	{
		this->x = other.x;
		this->y = other.y;
		this->z = other.z;
	}

	virtual ~Normal(){};

	/**
	 * @brief	Normalizes the components.
	 */
	void normalize();

	Normal<CoordType> operator+(const Normal &n) const
	{
	    return Normal<CoordType>(this->x + n.x, this->y + n.y, this->z + n.z);
	}

	virtual Normal<CoordType> operator-(const Normal &n) const
	{
		return Normal<CoordType>(this->x - n.x, this->y - n.y, this->z - n.z);
	}

	void operator+=(const Normal &n)
	{
		*this = *this + n;
	}

	void operator-=(const Normal &n)
	{
		*this = *this + n;

	}
};

typedef Normal<float> Normalf;

typedef Normal<double> Normald;

template<typename T>
inline ostream& operator<<(ostream& os, const Normal<T> &n)
{
	os << "Normal: " << n.x << " " << n.y << " " << n.z << endl;
	return os;
}

} // namespace lvr

#include "Normal.tcc"

#endif /* NORMAL_HPP_ */
