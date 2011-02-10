#ifndef __BASE_VERTEX_H__
#define __BASE_VERTEX_H__


#include <iostream>
#include <math.h>

using namespace std;


namespace lssr {

// Forward deklaration for matrix class
template<typename T> class Matrix4;

/**
 * @brief 		Basic vertex class. Supports all arithmetic operators
 * 				as well as indexed access and matrix multiplication
 * 				with the Matrix4 class
 */
template<typename CoordType>
class Vertex{

public:

	/**
	 * @brief	Default constructor. All coordinates are initialized
	 * 			with zeros.
	 */
	Vertex()
	{
		m_x = m_y = m_z = 0;
	}

	/**
	 * @brief	Builds a Vertex with the given coordinates.
	 */
	Vertex(const CoordType &x, const CoordType &y, const CoordType &z)
	{
		m_x = x;
		m_y = y;
		m_z = z;
	}

	/**
	 * @brief	Copy Ctor.
	 */
	Vertex(const Vertex &o)
	{
		m_x = o.m_x;
		m_y = o.m_y;
		m_z = o.m_z;
	}

	/**
	 * @brief	Destructor
	 */
	virtual ~Vertex(){};

	/**
	 * @brief 	Return the current length of the vector
	 */
	CoordType length()
	{
		return sqrt(m_x * m_x + m_y * m_y + m_z * m_z);
	}

	/**
	 * @brief	Calculates the cross product between this and
	 * 			the given vector. Returns a new Vertex instance.
	 *
	 * @param	other		The second cross product vector
	 */
	template<typename T>
	Vertex<CoordType> cross(const Vertex<T> &other) const
	{
		CoordType tx = m_y * other.m_z - m_z * other.m_y;
		CoordType ty = m_z * other.m_x - m_x * other.m_z;
		CoordType tz = m_x * other.m_y - m_y * other.m_x;

		return Vertex<CoordType>(tx, ty, tz);
    }


	/**
	 * @brief	Applies the given matrix. Translational components
	 * 			are ignored.
	 *
	 * @param	A 4x4 rotation matrix.
	 */
	template<typename T>
	void rotate(const Matrix4<T> &m)
	{
		m_x = m_x * m[0 ] + m_y * m[1 ] + m_z * m[2 ];
		m_y = m_x * m[4 ] + m_y * m[5 ] + m_z * m[6 ];
		m_z = m_x * m[8 ] + m_y * m[9 ] + m_z * m[10];

	}

	/**
	 * @brief	Calculates the cross product with the other given
	 * 			Vertex and assigns the result to the current
	 * 			instance.
	 */
	virtual void crossTo(const Vertex &other)
	{
		m_x = m_y * other.m_z - m_z * other.m_y;
		m_y = m_z * other.m_x - m_x * other.m_z;
		m_z = m_x * other.m_y - m_y * other.m_x;
	}


	/**
	 * @brief	Multiplication operator (dot product)
	 */
	virtual CoordType operator*(const Vertex &other) const
	{
		return m_x * other.m_x + m_y * other.m_y + m_z * other.m_z;
	}

	/**
	 * @brief 	Multiplication operator (scaling)
	 */
	virtual Vertex operator*(const CoordType &scale) const
	{
		return Vertex<CoordType>(m_x * scale, m_y * scale, m_z * scale);
	}


	virtual Vertex operator+(const Vertex &other) const
	{
		return Vertex(m_x + other.m_x, m_y + other.m_y, m_z + other.m_z);
	}

	/**
	 * @brief	Coordinate subtraction
	 */
	virtual Vertex operator-(const Vertex &other) const
	{
		return Vertex(m_x - other.m_x, m_y - other.m_y, m_z - other.m_z);
	}

	/**
	 * @brief	Coordinate substraction
	 */
	virtual void operator-=(const Vertex &other)
	{
		m_x -= other.m_x;
		m_y -= other.m_y;
		m_z -= other.m_z;
	}

	/**
	 * @brief	Coordinate addition
	 */
	virtual void operator+=(const Vertex &other)
	{
		m_x += other.m_x;
		m_y += other.m_y;
		m_z += other.m_z;
	}


	/**
	 * @brief	Scaling
	 */
	virtual void operator*=(const CoordType &scale)
		  {
		m_x *= scale;
		m_y *= scale;
		m_z *= scale;
	}

	/**
	 * @brief	Scaling
	 */
	virtual void operator/=(const CoordType &scale)
	{
		if(scale != 0)
		{
			m_x /= scale;
			m_y /= scale;
			m_z /= scale;
		}
		else
		{
			m_x = m_y = m_z = 0;
		}
	}

	/**
	 * @brief	Compares two vertices
	 */
	virtual bool operator==(const Vertex &other) const
	{
		return fabs(m_x - other.m_x) <= Vertex::epsilon &&
				fabs(m_y - other.m_y) <= Vertex::epsilon &&
				fabs(m_z - other.m_z) <= Vertex::epsilon;
	}

	/**
	 * @brief	Compares two vertices
	 */
	virtual bool operator!=(const Vertex &other) const
	{
		return !(*this == other);
	}

	/**
	 * @brief	Indexed coordinate access
	 */
	virtual CoordType operator[](const int &index) const
	{
		CoordType ret = 0.0;

		switch(index){
		case 0:
			ret = m_x;
			break;
		case 1:
			ret = m_y;
			break;
		case 2:
			ret = m_z;
			break;
		default:
			cout << "Vertex: Warning: Access index out of range." << endl;
		}
		return ret;
	}

	/// The x-coordinate of the vertex
	CoordType m_x;

	/// The y-coordinate of the vertex
	CoordType m_y;

	/// The z-coordinate of the vertex
	CoordType m_z;

private:

	/// Epsilon value for vertex comparism
	static const float epsilon = 0.001;
};


/**
 * @brief	Output operator for vertex types
 */
template<typename CoordType>
inline ostream& operator<<(ostream& os, const Vertex<CoordType> v){
	os << "Vertex: " << v.m_x << " " << v.m_y << " " << v.m_z << endl;
	return os;
}


/// Convenience typedef for float vertices
typedef Vertex<float> 	Vertexf;

/// Convenience typedef for double vertices
typedef Vertex<double>	Vertexd;

#include "Vertex.tcc"

} // namespace lssr

#endif
