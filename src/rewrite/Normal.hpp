/*
 * Normal.hpp
 *
 *  Created on: 04.02.2011
 *      Author: Thomas Wiemann
 */

#ifndef NORMAL_HPP_
#define NORMAL_HPP_

#include "Vertex.hpp"

namespace lssr {

/**
 * @brief	A normal implementation. Basically a vertex
 * 			with normalization functionality.
 */
template<typename CoordType>
class Normal : public Vertex<CoordType>
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
		this.m_x = other.m_x;
		this.m_y = other.m_y;
		this.m_z = other.m_z;
	}

	virtual ~Normal(){};

	/**
	 * @brief	Normalizes the components.
	 */
	void normalize()
	{
		//Don't normalize if we don't have to
		float l_square =
				  this->m_x * this->m_x
				+ this->m_y * this->m_y
				+ this->m_z * this->m_z;

		if( fabs(1 - l_square) > 0.001){

			float length = sqrt(l_square);
			if(length != 0){
				this->m_x /= length;
				this->m_y /= length;
				this->m_z /= length;
			}
		}
	}

	Normal<CoordType> operator+(const Normal &n) const
	{
	    return Normal<CoordType>(this->m_x + n.m_x, this->m_y + n.m_y, this->m_z + n.m_z);
	}

	virtual Normal<CoordType> operator-(const Normal &n) const
	{
		return Normal<CoordType>(this->m_x - n.m_x, this->m_y - n.m_y, this->m_z - n.m_z);
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

}
#endif /* NORMAL_HPP_ */
