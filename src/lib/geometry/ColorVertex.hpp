/*
 * ColorVertex.h
 *
 *  Created on: 08.09.2011
 *      Author: pg2011
 */

#ifndef COLORVERTEX_H_
#define COLORVERTEX_H_

using namespace std;

#include "Vertex.hpp"

namespace lssr
{

/**
 * @brief	A color vertex
 */
template<typename CoordType>
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
	}

	/**
	 * @brief	Builds a Vertex with the given coordinates.
	 */
	ColorVertex(const CoordType &_x, const CoordType &_y, const CoordType &_z, const uchar _r, const uchar _g, const uchar _b)
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

	uchar r, g, b;

};


/**
 * @brief	Output operator for color vertex types
 */
template<typename CoordType>
inline ostream& operator<<(ostream& os, const ColorVertex<CoordType> v){
	os << "ColorVertex: " << v.x << " " << v.y << " " << v.z << " " << (int)v.r << " " << (int)v.g << " " << (int)v.b << endl;
	return os;
}


} // namespace lssr

#endif /* COLORVERTEX_H_ */
