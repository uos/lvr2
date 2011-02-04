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
class BaseVertex{

public:

	/**
	 * @brief	Default constructor. All coordinates are initialized
	 * 			with zeros.
	 */
	BaseVertex()
	{
		m_x = m_y = m_z = 0;
	}

	/**
	 * @brief	Builds a Vertex with the given coordinates.
	 */
	BaseVertex(CoordType x, CoordType y, CoordType z)
	{
		m_x = x;
		m_y = y;
		m_z = z;
	}

	/**
	 * @brief	Copy Ctor.
	 */
	BaseVertex(const BaseVertex &o)
	{
		m_x = o.x;
		m_y = o.y;
		m_z = o.z;
	}

	/**
	 * @brief	Destructor
	 */
	virtual ~BaseVertex(){};

	/**
	 * @brief 	Return the current length of the vector
	 */
	CoordType length()
	{
		return sqrt(m_x * m_x + m_y * m_y + m_z * m_z);
	}

	/**
	 * @brief	Calculates the cross product between this and
	 * 			the given vector. Returns a new BaseVertex instance.
	 *
	 * @param	other		The second cross product vector
	 */
	template<typename T>
	BaseVertex<CoordType> cross(const BaseVertex<T> &other) const
	{
		CoordType tx = m_y * other.m_z - m_z * other.m_y;
		CoordType ty = m_z * other.m_x - m_x * other.m_z;
		CoordType tz = m_x * other.m_y - m_y * other.m_x;

		return BaseVertex<CoordType>(tx, ty, tz);
    }


	/**
	 * @brief	Applies the given matrix. Translational components
	 * 			are ignored.
	 *
	 * @param	A 4x4 rotation matrix.
	 */
	template<typename T>
	void rotate(Matrix4<T> m)
	{
		m_x = m_x * m[0 ] + m_y * m[1 ] + m_z * m[2 ];
		m_y = m_x * m[4 ] + m_y * m[5 ] + m_z * m[6 ];
		m_z = m_x * m[8 ] + m_y * m[9 ] + m_z * m[10];

	}

	/**
	 * @brief	Calculates the cross product with the other given
	 * 			BaseVertex and assigns the result to the current
	 * 			instance.
	 */
	virtual void crossTo(const BaseVertex &other)
	{
		m_x = m_y * other.z - m_z * other.y;
		m_y = m_z * other.x - m_x * other.z;
		m_z = m_x * other.y - m_y * other.x;
	}


	/**
	 * @brief	Multiplication operator (dot product)
	 */
	virtual CoordType operator*(const BaseVertex &other) const
	{
		return m_x * other.m_x + m_y * other.m_y + m_z * other.m_z;
	}

	/**
	 * @brief 	Multiplication operator (scaling)
	 */
	virtual BaseVertex operator*(const CoordType &scale) const
	{
		return BaseVertex<CoordType>(m_x * scale, m_y * scale, m_z * scale);
	}


	virtual BaseVertex operator+(const BaseVertex &other) const
	{
		return BaseVertex(m_x + other.m_x, m_y + other.m_y, m_z + other.m_z);
	}

	/**
	 * @brief	Coordinate subtraction
	 */
	virtual BaseVertex operator-(const BaseVertex &other) const
	{
		return BaseVertex(m_x - other.m_x, m_y - other.m_y, m_z - other.m_z);
	}

	/**
	 * @brief	Coordinate substraction
	 */
	virtual void operator-=(const BaseVertex &other)
	{
		m_x -= other.m_x;
		m_y -= other.m_y;
		m_z -= other.m_z;
	}

	/**
	 * @brief	Coordinate addition
	 */
	virtual void operator+=(const BaseVertex &other)
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
	virtual bool operator==(const BaseVertex &other) const
	{
		return fabs(m_x - other.x) <= BaseVertex::epsilon &&
				fabs(m_y - other.y) <= BaseVertex::epsilon &&
				fabs(m_z - other.z) <= BaseVertex::epsilon;
	}

	/**
	 * @brief	Compares two vertices
	 */
	virtual bool operator!=(const BaseVertex &other) const
	{
		return !(*this == other);
	}

	/**
	 * @brief	Indexed coordinate access
	 */
	virtual CoordType operator[](const int &index)
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
			cout << "BaseVertex: Warning: Access index out of range." << endl;
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
	static const float epsilon;
};


/**
 * @brief	Output operator for vertex types
 */
template<typename CoordType>
inline ostream& operator<<(ostream& os, const BaseVertex<CoordType> v){
	os << "BaseVertex: " << v.m_x << " " << v.m_y << " " << v.m_z << endl;
	return os;
}

/// Convenience typedef for float vertices
typedef BaseVertex<float> 	Vertexf;

/// Convenience typedef for double vertices
typedef BaseVertex<double>	Vertexd;

} // namespace lssr

#endif
