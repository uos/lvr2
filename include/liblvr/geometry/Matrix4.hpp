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
 * Matrix.hpp
 *
 *  @date 26.08.2008
 *  @author Thomas Wiemann (twiemann@uos.de)
 */

#ifndef MATRIX_H_
#define MATRIX_H_

#include <iostream>
#include <fstream>
#include <iomanip>

#include "Vertex.hpp"
#include "Normal.hpp"

#define _USE_MATH_DEFINES
#include <cmath>

using namespace std;

namespace lvr{

/**
 * @brief	A 4x4 matrix class implementation for use with the provided
 * 			vertex types.
 */
template<typename ValueType>
class Matrix4 {
public:

	/**
	 * @brief 	Default constructor. Initializes a identity matrix.
	 */
	Matrix4()
	{
		for(int i = 0; i < 16; i++) m[i] = 0;
		m[0] = m[5] = m[10] = m[15] = 1;
	}

	/**
	 * @brief	Initializes a matrix wit the given data array. Ensure
	 * 			that the array has exactly 16 fields.
	 */
	template<typename T>
	Matrix4(T* matrix)
	{
		for(int i = 0; i < 16; i++) m[i] = matrix[i];
	}

	/**
	 * @brief 	Copy constructor.
	 */
	template<typename T>
	Matrix4(const Matrix4<T>& other)
	{
		for(int i = 0; i < 16; i++) m[i] = other[i];
	}

	/**
	 * @brief	Constructs a matrix from given axis and angle. Trys to
	 * 			avoid a gimbal lock.
	 */
	template<typename T>
	Matrix4(Vertex<T> axis, ValueType angle)
	{
		// Check for gimbal lock
		if(fabs(angle) < 0.0001){

			bool invert_z = axis.z < 0;

			//Angle to yz-plane
			float pitch = atan2(axis.z, axis.x) - M_PI_2;
			if(pitch < 0.0f) pitch += 2.0f * M_PI;

			if(axis.x == 0.0f && axis.z == 0.0) pitch = 0.0f;

			//Transform axis into yz-plane
			axis.x =  axis.x * cos(pitch) + axis.z * sin(pitch);
			axis.z = -axis.x * sin(pitch) + axis.z * cos(pitch);

			//Angle to y-Axis
			float yaw = atan2(axis.y, axis.z);
			if(yaw < 0) yaw += 2 * M_PI;

			Matrix4 m1, m2, m3;

			if(invert_z) yaw = -yaw;

			cout << "YAW: " << yaw << " PITCH: " << pitch << endl;

			if(fabs(yaw)   > 0.0001){
				m2 = Matrix4(Vertex<T>(1.0, 0.0, 0.0), yaw);
				m3 = m3 * m2;
			}

			if(fabs(pitch) > 0.0001){
				m1 = Matrix4(Vertex<T>(0.0, 1.0, 0.0), pitch);
				m3 = m3 * m1;
			}

			for(int i = 0; i < 16; i++) m[i] = m3[i];

		} else {
			float c = cos(angle);
			float s = sin(angle);
			float t = 1.0f - c;
			float tmp1, tmp2;

			// Normalize axis
			Normal<T> a(axis);

			m[ 0] = c + a.x * a.x * t;
			m[ 5] = c + a.y * a.y * t;
			m[10] = c + a.z * a.z * t;

			tmp1 = a.x * a.y * t;
			tmp2 = a.z * s;
			m[ 4] = tmp1 + tmp2;
			m[ 1] = tmp1 - tmp2;

			tmp1 = a.x * a.z * t;
			tmp2 = a.y * s;
			m[ 8] = tmp1 - tmp2;
			m[ 2] = tmp1 + tmp2;

			tmp1 = a.y * a.z * t;
			tmp2 = a.x * s;
			m[ 9] = tmp1 + tmp2;
			m[ 6] = tmp1 - tmp2;

			m[ 3] = m[ 7] = m[11] = 0.0;
			m[12] = m[13] = m[14] = 0.0;
			m[15] = 1.0;
		}
	}

	template<typename T>
	Matrix4(const Vertex<T> &position, const Vertex<T> &angles)
	{
		float sx = sin(angles[0]);
		float cx = cos(angles[0]);
		float sy = sin(angles[1]);
		float cy = cos(angles[1]);
		float sz = sin(angles[2]);
		float cz = cos(angles[2]);

		m[0]  = cy*cz;
		m[1]  = sx*sy*cz + cx*sz;
		m[2]  = -cx*sy*cz + sx*sz;
		m[3]  = 0.0;
		m[4]  = -cy*sz;
		m[5]  = -sx*sy*sz + cx*cz;
		m[6]  = cx*sy*sz + sx*cz;
		m[7]  = 0.0;
		m[8]  = sy;
		m[9]  = -sx*cy;
		m[10] = cx*cy;

		m[11] = 0.0;

		m[12] = position[0];
		m[13] = position[1];
		m[14] = position[2];
		m[15] = 1;
	}

	Matrix4(string filename);

	virtual ~Matrix4()
	{

	}

	/**
	 * @brief	Scales the matrix elemnts by the given factor
	 */
	template<typename T>
	Matrix4 operator*(const T &scale) const
	{
		ValueType new_matrix[16];
		for(int i = 0; i < 16; i++){
			new_matrix[i] = m[i] * scale;
		}
		return Matrix4<ValueType>(new_matrix);
	}

	/**
	 * @brief	Matrix-Matrix multiplication. Returns the new
	 * 			matrix
	 */
	template<typename T>
	Matrix4 operator*(const Matrix4<T> &other) const
	{
		ValueType new_matrix[16];
		new_matrix[ 0] = m[ 0] * other[ 0] + m[ 4] * other[ 1] + m[ 8] * other[ 2] + m[12] * other[ 3];
		new_matrix[ 1] = m[ 1] * other[ 0] + m[ 5] * other[ 1] + m[ 9] * other[ 2] + m[13] * other[ 3];
		new_matrix[ 2] = m[ 2] * other[ 0] + m[ 6] * other[ 1] + m[10] * other[ 2] + m[14] * other[ 3];
		new_matrix[ 3] = m[ 3] * other[ 0] + m[ 7] * other[ 1] + m[11] * other[ 2] + m[15] * other[ 3];
		new_matrix[ 4] = m[ 0] * other[ 4] + m[ 4] * other[ 5] + m[ 8] * other[ 6] + m[12] * other[ 7];
		new_matrix[ 5] = m[ 1] * other[ 4] + m[ 5] * other[ 5] + m[ 9] * other[ 6] + m[13] * other[ 7];
		new_matrix[ 6] = m[ 2] * other[ 4] + m[ 6] * other[ 5] + m[10] * other[ 6] + m[14] * other[ 7];
		new_matrix[ 7] = m[ 3] * other[ 4] + m[ 7] * other[ 5] + m[11] * other[ 6] + m[15] * other[ 7];
		new_matrix[ 8] = m[ 0] * other[ 8] + m[ 4] * other[ 9] + m[ 8] * other[10] + m[12] * other[11];
		new_matrix[ 9] = m[ 1] * other[ 8] + m[ 5] * other[ 9] + m[ 9] * other[10] + m[13] * other[11];
		new_matrix[10] = m[ 2] * other[ 8] + m[ 6] * other[ 9] + m[10] * other[10] + m[14] * other[11];
		new_matrix[11] = m[ 3] * other[ 8] + m[ 7] * other[ 9] + m[11] * other[10] + m[15] * other[11];
		new_matrix[12] = m[ 0] * other[12] + m[ 4] * other[13] + m[ 8] * other[14] + m[12] * other[15];
		new_matrix[13] = m[ 1] * other[12] + m[ 5] * other[13] + m[ 9] * other[14] + m[13] * other[15];
		new_matrix[14] = m[ 2] * other[12] + m[ 6] * other[13] + m[10] * other[14] + m[14] * other[15];
		new_matrix[15] = m[ 3] * other[12] + m[ 7] * other[13] + m[11] * other[14] + m[15] * other[15];
		return Matrix4<ValueType>(new_matrix);
	}

	/**
	 * @brief 	Matrix addition operator. Returns a new matrix
	 *
	 */
	template<typename T>
	Matrix4 operator+(const Matrix4<T> &other) const
	{
		ValueType new_matrix[16];
		for(int i = 0; i < 16; i++)
		{
			new_matrix[i] = m[i] + other[i];
		}
		return Matrix4<ValueType>(new_matrix);
	}

	/**
	 * @brief 	Matrix addition operator
	 */
	template<typename T>
	Matrix4 operator+=(const Matrix4<T> &other)
	{
		if(other != *this)
		{
			return *this + other;
		}
		else
		{
			return *this;
		}
	}

	/**
	 * @brief	Matrix-Matrix multiplication (array based). Mainly
	 * 			implemented for compatibility with other math libs.
	 * 			ensure that the used array has at least 16 elements
	 * 			to avoid memory access violations.
	 */
	template<typename T>
	Matrix4 operator*(const T* &other) const
	{
		ValueType new_matrix[16];
		new_matrix[ 0] = m[ 0] * other[ 0] + m[ 4] * other[ 1] + m[ 8] * other[ 2] + m[12] * other[ 3];
		new_matrix[ 1] = m[ 1] * other[ 0] + m[ 5] * other[ 1] + m[ 9] * other[ 2] + m[13] * other[ 3];
		new_matrix[ 2] = m[ 2] * other[ 0] + m[ 6] * other[ 1] + m[10] * other[ 2] + m[14] * other[ 3];
		new_matrix[ 3] = m[ 3] * other[ 0] + m[ 7] * other[ 1] + m[11] * other[ 2] + m[15] * other[ 3];
		new_matrix[ 4] = m[ 0] * other[ 4] + m[ 4] * other[ 5] + m[ 8] * other[ 6] + m[12] * other[ 7];
		new_matrix[ 5] = m[ 1] * other[ 4] + m[ 5] * other[ 5] + m[ 9] * other[ 6] + m[13] * other[ 7];
		new_matrix[ 6] = m[ 2] * other[ 4] + m[ 6] * other[ 5] + m[10] * other[ 6] + m[14] * other[ 7];
		new_matrix[ 7] = m[ 3] * other[ 4] + m[ 7] * other[ 5] + m[11] * other[ 6] + m[15] * other[ 7];
		new_matrix[ 8] = m[ 0] * other[ 8] + m[ 4] * other[ 9] + m[ 8] * other[10] + m[12] * other[11];
		new_matrix[ 9] = m[ 1] * other[ 8] + m[ 5] * other[ 9] + m[ 9] * other[10] + m[13] * other[11];
		new_matrix[10] = m[ 2] * other[ 8] + m[ 6] * other[ 9] + m[10] * other[10] + m[14] * other[11];
		new_matrix[11] = m[ 3] * other[ 8] + m[ 7] * other[ 9] + m[11] * other[10] + m[15] * other[11];
		new_matrix[12] = m[ 0] * other[12] + m[ 4] * other[13] + m[ 8] * other[14] + m[12] * other[15];
		new_matrix[13] = m[ 1] * other[12] + m[ 5] * other[13] + m[ 9] * other[14] + m[13] * other[15];
		new_matrix[14] = m[ 2] * other[12] + m[ 6] * other[13] + m[10] * other[14] + m[14] * other[15];
		new_matrix[15] = m[ 3] * other[12] + m[ 7] * other[13] + m[11] * other[14] + m[15] * other[15];
		return Matrix4<ValueType>(new_matrix);
	}

	/**
	 * @brief	Multiplication of Matrix and Vertex types
	 */
	template<typename T>
	Vertex<T> operator*(const Vertex<T> &v) const
	{
		T x = m[ 0] * v.x + m[ 4] * v.y + m[8 ] * v.z;
		T y = m[ 1] * v.x + m[ 5] * v.y + m[9 ] * v.z;
		T z = m[ 2] * v.x + m[ 6] * v.y + m[10] * v.z;

		x = x + m[12];
		y = y + m[13];
		z = z + m[14];

		return Vertex<T>(x, y, z);
	}

	/**
	 * @brief	Sets the given index of the Matrix's data field
	 * 			to the provided value.
	 *
	 * @param	i		Field index of the matrix
	 * @param	value	new value
	 */
	void set(int i, ValueType value){m[i] = value;};

	/**
	 * @brief	Transposes the current matrix
	 */
	void transpose()
	{
		ValueType m_tmp[16];
		m_tmp[0]  = m[0];
		m_tmp[4]  = m[1];
		m_tmp[8]  = m[2];
		m_tmp[12] = m[3];
		m_tmp[1]  = m[4];
		m_tmp[5]  = m[5];
		m_tmp[9]  = m[6];
		m_tmp[13] = m[7];
		m_tmp[2]  = m[8];
		m_tmp[6]  = m[9];
		m_tmp[10] = m[10];
		m_tmp[14] = m[11];
		m_tmp[3]  = m[12];
		m_tmp[7]  = m[13];
		m_tmp[11] = m[14];
		m_tmp[15] = m[15];
		for(int i = 0; i < 16; i++) m[i] = m_tmp[i];
	}

	/**
	 * @brief	Computes an Euler representation (x, y, z) plus three
	 * 			rotation values in rad. Rotations are with respect to
	 * 			the x, y, z axes.
	 */
	void toPostionAngle(ValueType pose[6])
	{
		if(pose != 0){
			float _trX, _trY;
			if(m[0] > 0.0) {
				pose[4] = asin(m[8]);
			} else {
				pose[4] = M_PI - asin(m[8]);
			}
			// rPosTheta[1] =  asin( m[8]);      // Calculate Y-axis angle

			float  C    =  cos( pose[4] );
			if ( fabs( C ) > 0.005 )  {          // Gimball lock?
				_trX      =  m[10] / C;          // No, so get X-axis angle
				_trY      =  -m[9] / C;
				pose[3]  = atan2( _trY, _trX );
				_trX      =  m[0] / C;           // Get Z-axis angle
				_trY      = -m[4] / C;
				pose[5]  = atan2( _trY, _trX );
			} else {                             // Gimball lock has occurred
				pose[3] = 0.0;                   // Set X-axis angle to zero
				_trX      =  m[5];  //1          // And calculate Z-axis angle
				_trY      =  m[1];  //2
				pose[5]  = atan2( _trY, _trX );
			}

			pose[0] = m[12];
			pose[1] = m[13];
			pose[2] = m[14];
		}
	}

	/**
	 * @brief	Loads matrix values from a given file.
	 */
	void loadFromFile(string filename)
	{
		ifstream in(filename.c_str());
		for(int i = 0; i < 16; i++){
			if(!in.good()){
				cout << "Warning: Matrix::loadFromFile: File not found or corrupted." << endl;
				return;
			}
			in >> m[i];
		}
	}

	/**
	 * @brief	Matrix scaling with self assignment.
	 */
	template<typename T>
	void operator*=(const T scale)
	{
		*this = *this * scale;
	}

	/**
	 * @brief 	Matrix-Matrix multiplication with self assigment.
	 */
	template<typename T>
	void operator*=(const Matrix4<T>& other)
	{
		*this = *this * other;
	}

	/**
	 * @brief	Matrix-Matrix multiplication (array based). See \ref{operator*}.
	 */
	template<typename T>
	void operator*=(const T* other)
	{
		*this = *this * other;
	}

	/**
	 * @brief	Returns the internal data array. Unsafe. Will probably
	 * 			removed in one of the next versions.
	 */
	ValueType* getData(){ return m;};

	/**
	 * @brief	Returns the element at the given index.
	 */
	ValueType at(const int i) const;

	/**
	 * @brief	Indexed element (reading) access.
	 */
	ValueType operator[](const int index) const
	{
	    /// TODO: Boundary check
	    return m[index];
	}


	/**
	 * @brief  	Writeable index access
	 */
	ValueType& operator[](const int index)
	{
		return m[index];
	}

	/**
	 * @brief   Returns the matrix's determinant
	 */
	ValueType& det()
	{
	    ValueType det, result = 0, i = 1.0;
	    ValueType Msub3[9];
	    int    n;
	    for ( n = 0; n < 4; n++, i *= -1.0 ) {
	        submat( Msub3, 0, n );
	        det     = det3( Msub3 );
	        result += m[n] * det * i;
	    }
	    return( result );
	}

	Matrix4 inv(Matrix4 in, bool &success = true)
	{
	    Matrix4 Mout;
	    ValueType  mdet = det();
	    if ( fabs( mdet ) < 0.00000000000005 ) {
	        cout << "Error matrix inverting! " << mdet << endl;
	        success = false;
	        return Mout;
	    }
	    ValueType  mtemp[9];
	    int     i, j, sign;
	    for ( i = 0; i < 4; i++ ) {
	        for ( j = 0; j < 4; j++ ) {
	            sign = 1 - ( (i +j) % 2 ) * 2;
	            submat( mtemp, i, j );
	            Mout[i+j*4] = ( det3( mtemp ) * sign ) / mdet;
	        }
	    }
	    success = true;
	    return Mout;
	}

private:

    /**
     * @brief   Returns a sub matrix without row \ref i and column \ref j.
     */
	void submat(ValueType* submat, int i, int j)
	{
	    int di, dj, si, sj;
	    // loop through 3x3 submatrix
	    for( di = 0; di < 3; di ++ ) {
	        for( dj = 0; dj < 3; dj ++ ) {
	            // map 3x3 element (destination) to 4x4 element (source)
	            si = di + ( ( di >= i ) ? 1 : 0 );
	            sj = dj + ( ( dj >= j ) ? 1 : 0 );
	            // copy element
	            submat[di * 3 + dj] = m[si * 4 + sj];
	        }
	    }
	}

	/**
	 * @brief    Calculates the determinant of a 3x3 matrix
	 *
	 * @param    M  input 3x3 matrix
	 * @return   determinant of input matrix
	 */
	ValueType det3( const ValueType *M )
	{
	  ValueType det;
	  det = (double)(  M[0] * ( M[4]*M[8] - M[7]*M[5] )
	                 - M[1] * ( M[3]*M[8] - M[6]*M[5] )
	                 + M[2] * ( M[3]*M[7] - M[6]*M[4] ));
	  return ( det );
	}

	ValueType m[16];
};

/**
 * @brief Output operator for matrices.
 */
template<typename T>
inline ostream& operator<<(ostream& os, const Matrix4<T> matrix){
	os << "Matrix:" << endl;
	os << fixed;
	for(int i = 0; i < 16; i++){
		os << setprecision(4) << matrix[i] << " ";
		if(i % 4 == 3) os << " " <<  endl;
	}
	os << endl;
	return os;
}

typedef Matrix4<float> Matrix4f;

} // namespace lvr
#endif /* MATRIX_H_ */
