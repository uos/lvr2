// ******************************
// vec3.h
//
// Vector class
// 
// Jeff Somers
// Copyright (c) 2002
//
// jsomers@alumni.williams.edu
// March 27, 2002
// ******************************

#ifndef __Vec3_h
#define __Vec3_h

#if defined (_MSC_VER) && (_MSC_VER >= 1020)
#pragma once
#pragma warning(disable:4710) // function not inlined
#pragma warning(disable:4702) // unreachable code
#pragma warning(disable:4514) // unreferenced inline function has been removed
#endif

#include <math.h>
#include <iostream>

// Vec3 is a vector class
// I couldn't call it vec3, because that already
// seems to be defined.
class Vec3 
{
public:
	float x, y, z;

	enum {MAX_INPUT_STRING = 40};

	// Constructors and Destructors
	Vec3() {x = y = z = 0.0f;};
	Vec3(float x1, float y1, float z1) {x = x1; y = y1; z = z1;};
	Vec3(float av[3]) {x = av[0]; y = av[1]; z = av[2];};
	Vec3(const Vec3& v) {x = v.x; y = v.y; z = v.z;};
	~Vec3() {}; // Destructor intentially does nothing

	// Assignment operator
	Vec3& operator=(const Vec3& v) {x = v.x; y = v.y; z = v.z; return *this;};

	// Comparision operators
	bool operator==(const Vec3& v) {return (x == v.x && y == v.y && z == v.z);};
	bool operator!=(const Vec3& v) {return (x != v.x || y != v.y || z != v.z);};

	// Scalar operations
	Vec3 operator+(float f) const {return Vec3(x + f, y + f, z + f);};
	Vec3 operator-(float f) const {return Vec3(x - f, y - f, z - f);};
	Vec3 operator*(float f) const {return Vec3(x * f, y * f, z * f);};
	Vec3 operator/(float f) const {Vec3 v1(x,y,z); if (f != 0.0f) {v1.x /= f; v1.y /= f; v1.z /= f;}; return v1;};

	Vec3& operator+=(float f) {x += f; y += f; z += f; return *this;};
	Vec3& operator-=(float f) {x -= f; y -= f; z -= f; return *this;};
	Vec3& operator*=(float f) {x *= f; y *= f; z *= f; return *this;};
	Vec3& operator/=(float f) {if(f!=0.0f){ x /= f; y /= f; z /= f;}; return *this;};


	// Vector operations
	Vec3 operator+(const Vec3& v) const {return Vec3(x + v.x, y + v.y, z + v.z);};
	Vec3& operator+=(const Vec3& v) {x += v.x; y += v.y; z += v.z; return *this;};
	Vec3 operator-(const Vec3& v) const {return Vec3(x - v.x, y - v.y, z - v.z);};
	Vec3& operator-=(const Vec3& v) {x -= v.x; y -= v.y; z -= v.z; return *this;};

	// Unary operators
	Vec3 operator-() const {return Vec3 (-x, -y, -z); };

	// Dot and Cross Products
	float dot(const Vec3& v) const {return (x * v.x + y * v.y + z * v.z);};
	Vec3 cross(const Vec3& v) const {return Vec3(y * v.z - z * v.y,
											 z * v.x - x * v.z,
											 x * v.y - y * v.x);};
	Vec3 unitcross(const Vec3& v) const {Vec3 vr(y * v.z - z * v.y,
											 z * v.x - x * v.z,
											 x * v.y - y * v.x); vr.normalize(); return vr;};

	// Miscellaneous
	void normalize() {float a = float(sqrt(x*x + y*y + z*z)); if (a!=0.0f) {x/=a; y/=a; z/=a;};};
	void setZero() {x = y = z = 0.0f;};
	float length() {return float(sqrt(x*x + y*y + z*z));};

	// Friend functions
	friend Vec3 operator*(float a, const Vec3& v) {return Vec3 (a * v.x, a * v.y, a * v.z);};

	// dot and cross products
	float dot(const Vec3& v1, const Vec3& v2) {return (v1.x * v2.x + v1.y * v2.y +v1. z * v2.z);};
	Vec3 cross(const Vec3& v1, const Vec3& v2)  {return Vec3 (v1.y * v2.z - v1.z * v2.y,
														v1.z * v2.x - v1.x * v2.z,
															v1.x * v2.y - v1.y * v2.x);};
	Vec3 unitcross(const Vec3& v1, const Vec3& v2)  {Vec3 vr(v1.y * v2.z - v1.z * v2.y,
													v1.z * v2.x - v1.x * v2.z,
													v1.x * v2.y - v1.y * v2.x); 
													vr.normalize(); return vr;};


	// Input and Output
	friend std::ostream& operator<<(std::ostream& os, const Vec3& vo);

//	friend istream& operator>>(istream& is, Vec3& vi);

private:
};

#endif // #ifndef __Vec3_h
