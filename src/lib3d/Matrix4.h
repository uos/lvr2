/*
 * Matrix.h
 *
 *  Created on: 26.08.2008
 *      Author: Thomas Wiemann
 */


#ifndef MATRIX_H_
#define MATRIX_H_

#include <iostream>
#include <fstream>
#include <iomanip>

#include "Constants.h"
#include "Normal.h"

using namespace std;

class Matrix4 {
public:
	Matrix4();
	Matrix4(double* matrix);
	Matrix4(float* matrix);
	Matrix4(const Matrix4& other);
	Matrix4(Vertex axis, float angle);
	Matrix4(Vertex position, Vertex angles);
	Matrix4(string filename);

	virtual ~Matrix4();

	virtual inline Matrix4 operator*(const double scale) const;
	virtual inline Matrix4 operator*(const Matrix4& other) const;
	virtual inline Matrix4 operator*(const double* other) const;

	virtual inline BaseVertex operator*(const BaseVertex v) const;

	void set(int i, float f){m[i] = f;};
	void transpose();
	void toPostionAngle(double* pose);
	void loadFromFile(string filename);

	virtual void operator*=(const double scale);
	virtual void operator*=(const Matrix4& other);
	virtual void operator*=(const double* other);

	double* getData(){ return m;};
	float at(const int i) const;

	float operator[](const int index) const;
		private:
	double m[16];
};

inline BaseVertex Matrix4::operator*(const BaseVertex v) const{
	float x = m[ 0] * v.x + m[ 4] * v.y + m[8 ] * v.z;
	float y = m[ 1] * v.x + m[ 5] * v.y + m[9 ] * v.z;
	float z = m[ 2] * v.x + m[ 6] * v.y + m[10] * v.z;

	x = x + m[12];
	y = y + m[13];
	z = z + m[14];

	return BaseVertex(x, y, z);
}


Matrix4 Matrix4::operator*(const Matrix4& other) const{
	double new_matrix[16];
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
	return Matrix4(new_matrix);
}

Matrix4 Matrix4::operator*(const double* other) const{
	double new_matrix[16];
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
	return Matrix4(new_matrix);
}

Matrix4 Matrix4::operator*(const double scale) const{
	double new_matrix[16];
	for(int i = 0; i < 16; i++){
		new_matrix[i] = m[i] * scale;
	}
	return Matrix4(new_matrix);
}



inline void Matrix4::operator*=(const Matrix4& other){
	m[ 0] = m[ 0] * other[ 0] + m[ 4] * other[ 1] + m[ 8] * other[ 2] + m[12] * other[ 3];
	m[ 1] = m[ 1] * other[ 0] + m[ 5] * other[ 1] + m[ 9] * other[ 2] + m[13] * other[ 3];
	m[ 2] = m[ 2] * other[ 0] + m[ 6] * other[ 1] + m[10] * other[ 2] + m[14] * other[ 3];
	m[ 3] = m[ 3] * other[ 0] + m[ 7] * other[ 1] + m[11] * other[ 2] + m[15] * other[ 3];
	m[ 4] = m[ 0] * other[ 4] + m[ 4] * other[ 5] + m[ 8] * other[ 6] + m[12] * other[ 7];
	m[ 5] = m[ 1] * other[ 4] + m[ 5] * other[ 5] + m[ 9] * other[ 6] + m[13] * other[ 7];
	m[ 6] = m[ 2] * other[ 4] + m[ 6] * other[ 5] + m[10] * other[ 6] + m[14] * other[ 7];
	m[ 7] = m[ 3] * other[ 4] + m[ 7] * other[ 5] + m[11] * other[ 6] + m[15] * other[ 7];
	m[ 8] = m[ 0] * other[ 8] + m[ 4] * other[ 9] + m[ 8] * other[10] + m[12] * other[11];
	m[ 9] = m[ 1] * other[ 8] + m[ 5] * other[ 9] + m[ 9] * other[10] + m[13] * other[11];
	m[10] = m[ 2] * other[ 8] + m[ 6] * other[ 9] + m[10] * other[10] + m[14] * other[11];
	m[11] = m[ 3] * other[ 8] + m[ 7] * other[ 9] + m[11] * other[10] + m[15] * other[11];
	m[12] = m[ 0] * other[12] + m[ 4] * other[13] + m[ 8] * other[14] + m[12] * other[15];
	m[13] = m[ 1] * other[12] + m[ 5] * other[13] + m[ 9] * other[14] + m[13] * other[15];
	m[14] = m[ 2] * other[12] + m[ 6] * other[13] + m[10] * other[14] + m[14] * other[15];
	m[15] = m[ 3] * other[12] + m[ 7] * other[13] + m[11] * other[14] + m[15] * other[15];
}

inline void Matrix4::operator*=(const double* other){
	m[ 0] = m[ 0] * other[ 0] + m[ 4] * other[ 1] + m[ 8] * other[ 2] + m[12] * other[ 3];
	m[ 1] = m[ 1] * other[ 0] + m[ 5] * other[ 1] + m[ 9] * other[ 2] + m[13] * other[ 3];
	m[ 2] = m[ 2] * other[ 0] + m[ 6] * other[ 1] + m[10] * other[ 2] + m[14] * other[ 3];
	m[ 3] = m[ 3] * other[ 0] + m[ 7] * other[ 1] + m[11] * other[ 2] + m[15] * other[ 3];
	m[ 4] = m[ 0] * other[ 4] + m[ 4] * other[ 5] + m[ 8] * other[ 6] + m[12] * other[ 7];
	m[ 5] = m[ 1] * other[ 4] + m[ 5] * other[ 5] + m[ 9] * other[ 6] + m[13] * other[ 7];
	m[ 6] = m[ 2] * other[ 4] + m[ 6] * other[ 5] + m[10] * other[ 6] + m[14] * other[ 7];
	m[ 7] = m[ 3] * other[ 4] + m[ 7] * other[ 5] + m[11] * other[ 6] + m[15] * other[ 7];
	m[ 8] = m[ 0] * other[ 8] + m[ 4] * other[ 9] + m[ 8] * other[10] + m[12] * other[11];
	m[ 9] = m[ 1] * other[ 8] + m[ 5] * other[ 9] + m[ 9] * other[10] + m[13] * other[11];
	m[10] = m[ 2] * other[ 8] + m[ 6] * other[ 9] + m[10] * other[10] + m[14] * other[11];
	m[11] = m[ 3] * other[ 8] + m[ 7] * other[ 9] + m[11] * other[10] + m[15] * other[11];
	m[12] = m[ 0] * other[12] + m[ 4] * other[13] + m[ 8] * other[14] + m[12] * other[15];
	m[13] = m[ 1] * other[12] + m[ 5] * other[13] + m[ 9] * other[14] + m[13] * other[15];
	m[14] = m[ 2] * other[12] + m[ 6] * other[13] + m[10] * other[14] + m[14] * other[15];
	m[15] = m[ 3] * other[12] + m[ 7] * other[13] + m[11] * other[14] + m[15] * other[15];
}

inline void Matrix4::operator*=(const double scale){
	for(int i = 0; i < 16; i++) m[i] = m[i] * scale;
}

inline ostream& operator<<(ostream& os, const Matrix4 matrix){
	os << "Matrix:" << endl;
	os << fixed;
	for(int i = 0; i < 16; i++){
		os << setprecision(4) << matrix.at(i) << " ";
		if(i % 4 == 3) os << " " <<  endl;
	}
	os << endl;
	return os;
}

#endif /* MATRIX_H_ */
