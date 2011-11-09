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


 #ifndef __BASE_VERTEX_HPP__
#define __BASE_VERTEX_HPP__


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
		x = y = z = 0;
	}

	/**
	 * @brief	Builds a Vertex with the given coordinates.
	 */
	Vertex(const CoordType &_x, const CoordType &_y, const CoordType &_z)
	{
		x = _x;
		y = _y;
		z = _z;
	}

	/**
	 * @brief	Copy Ctor.
	 */
	Vertex(const Vertex &o)
	{
		x = o.x;
		y = o.y;
		z = o.z;
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
	 * 			are ignored (matrix must be row major).
	 *
	 * @param	A 4x4 rotation matrix.
	 */
	void rotate(const Matrix4<CoordType> &m);

    /**
     * @brief   Applies the given matrix. Translational components
     *          are ignored
     *
     * @param   A 4x4 rotation matrix (column major).
     */
    void rotateCM(const Matrix4<CoordType> &m);

    /**
     * @brief   Applies the given matrix. Translational components
     *          are ignored
     *
     * @param   A 4x4 rotation matrix (row major).
     */
    void rotateRM(const Matrix4<CoordType> &m);

	/**
	 * @brief	Transforms the vertex according to the given matrix
	 *          (Default: matrix is row major)
	 *
	 * @param	A 4x4 tranformation matrix.
	 */
	void transform(const Matrix4<CoordType> &m);

	/**
	 * @brief   Transforms the vertex according to the given matrix
	 *          (Row major format)
	 *
	 * @param   A 4x4 tranformation matrix.
	 */
	void transformRM(const Matrix4<CoordType> &m);

    /**
     * @brief   Transforms the vertex according to the given matrix
     *          (Column major format)
     *
     * @param   A 4x4 tranformation matrix.
     */
    void transformCM(const Matrix4<CoordType> &m);




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
	virtual void operator+=(const Vertex<CoordType> &other);


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


  virtual bool operator<(const Vertex &other) const
  {
     if( this->x < other.x ) return true;
     if( (this->x - other.x) <= 0.00001 )
     {
        if(this->y < other.y ) return true;
        if((this->y - other.y) <= 0.00001 ) 
           if(this->z < other.z) return true;
     }
     return false;
  }

	/**
	 * @brief	Indexed coordinate access (reading)
	 */
	virtual CoordType operator[](const int &index) const;

    /**
     * @brief   Indexed coordinate access (writing)
     */
    virtual CoordType& operator[](const int &index);


	/// The x-coordinate of the vertex
	CoordType x;

	/// The y-coordinate of the vertex
	CoordType y;

	/// The z-coordinate of the vertex
	CoordType z;

private:

	/// Epsilon value for vertex comparism
	static const float epsilon = 0.00001;
};


/**
 * @brief	Output operator for vertex types
 */
template<typename CoordType>
inline ostream& operator<<(ostream& os, const Vertex<CoordType> v){
	os << "Vertex: " << v.x << " " << v.y << " " << v.z << endl;
	return os;
}


/// Convenience typedef for float vertices
typedef Vertex<float> 	Vertexf;

/// Convenience typedef for double vertices
typedef Vertex<double>	Vertexd;

} // namespace lssr

#include "Vertex.tcc"

#endif
