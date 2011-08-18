/*
 * Normal.hpp
 *
 *  Created on: 04.02.2011
 *      Author: Thomas Wiemann
 */

#ifndef NORMAL_HPP_
#define NORMAL_HPP_


#include "./Vertex.hpp"

namespace lssr {

/**
 * @brief	A normal implementation. Basically a vertex
 * 			with normalization functionality.
 */
template<typename CoordType>
class Normal : public lssr::Vertex<CoordType>
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
		this.x = other.x;
		this.y = other.y;
		this.z = other.z;
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

} // namespace lssr

#include "Normal.tcc"

#endif /* NORMAL_HPP_ */
