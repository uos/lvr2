/*
 * Normal.hpp
 *
 *  Created on: 04.02.2011
 *      Author: Thomas Wiemann
 */

#ifndef NORMAL_HPP_
#define NORMAL_HPP_

#include "BaseVertex.hpp"

namespace lssr {

/**
 * @brief	A normal implementation. Basically a vertex
 * 			with normalization functionality.
 */
template<typename CoordType>
class Normal : public BaseVertex<CoordType>
{

public:

	/**
	 * @brief	Default constructor. All elements are set
	 * 			to zero.
	 */
	Normal() : BaseVertex<CoordType>() {};

	/**
	 * @brief	Constructs a new normal from the given
	 * 			components. Applies normalization.
	 */
	Normal(CoordType x, CoordType y, CoordType z)
	: BaseVertex<CoordType>(x, y, z)
	  {
		normalize();
	  }

	/**
	 * @brief	Copy constructor for vertices. Applies
	 * 			normalization.
	 */
	template<typename T>
	Normal(const BaseVertex<T> &other) : BaseVertex<T>(other)
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
				  this.m_x * this.x
				+ this.y * this.y
				+ this.z * this.z;

		if( fabs(1 - l_square) > 0.001){

			float length = sqrt(l_square);
			if(length != 0){
				this.m_x /= length;
				this.m_y /= length;
				this.m_z /= length;
			}
		}
	}

	Normal operator+(const Normal &n) const
	{
		Normal res = ::operator+(n);
		return res.normalize();
	}

	virtual Normal operator-(const Normal &n) const
	{
		Normal res = ::operator+(n);
		return res.normalize();
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

template<typename T>
inline ostream& operator<<(ostream& os, const Normal<T> &n)
{
	os << "Normal: " << n.x << " " << n.y << " " << n.z << endl;
	return os;
}

}
#endif /* NORMAL_HPP_ */
