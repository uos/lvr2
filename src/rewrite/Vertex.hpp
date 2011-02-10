#ifndef __BASE_VERTEX_H__
#define __BASE_VERTEX_H__


#include <iostream>
#include <math.h>

using namespace std;


namespace lssr
{

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
	CoordType length();

	/**
	 * @brief	Calculates the cross product between this and
	 * 			the given vector. Returns a new Vertex instance.
	 *
	 * @param	other		The second cross product vector
	 */
	Vertex<CoordType> cross(const Vertex &other) const;


	/**
	 * @brief	Applies the given matrix. Translational components
	 * 			are ignored.
	 *
	 * @param	A 4x4 rotation matrix.
	 */
	void rotate(const Matrix4<CoordType> &m);

	/**
	 * @brief	Calculates the cross product with the other given
	 * 			Vertex and assigns the result to the current
	 * 			instance.
	 */
	virtual void crossTo(const Vertex &other);

	/**
	 * @brief	Multiplication operator (dot product)
	 */
	virtual CoordType operator*(const Vertex &other) const;


	/**
	 * @brief 	Multiplication operator (scaling)
	 */
	virtual Vertex<CoordType> operator*(const CoordType &scale) const;

	virtual Vertex<CoordType> operator+(const Vertex &other) const;


	/**
	 * @brief	Coordinate subtraction
	 */
	virtual Vertex<CoordType> operator-(const Vertex &other) const;

	/**
	 * @brief	Coordinate substraction
	 */
	virtual void operator-=(const Vertex &other);

	/**
	 * @brief	Coordinate addition
	 */
	virtual void operator+=(const Vertex &other);


	/**
	 * @brief	Scaling
	 */
	virtual void operator*=(const CoordType &scale);


	/**
	 * @brief	Scaling
	 */
	virtual void operator/=(const CoordType &scale);

	/**
	 * @brief	Compares two vertices
	 */
	virtual bool operator==(const Vertex &other) const;

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
	virtual CoordType operator[](const int &index) const;


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

} // namespace lssr

#include "Vertex.tcc"

#endif
