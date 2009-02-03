// ******************************
// vec3.cpp
//
// Vector class
// 
// Jeff Somers
// Copyright (c) 2002
//
// jsomers@alumni.williams.edu
// March 27, 2002
// ******************************

// Friend functions for input/output

#if defined (_MSC_VER) && (_MSC_VER >= 1020)
#pragma warning(disable:4710) // function not inlined
#pragma warning(disable:4702) // unreachable code
#pragma warning(disable:4514) // unreferenced inline function has been removed
#endif

#include "vec3.h"

std::ostream&
operator<<(std::ostream& os, const Vec3& vo)
{
	return os << "<" << vo.x << ", " << vo.y << ", " << vo.z << ">";
}

/* NOT IMPLEMENTED
istream& operator>>(istream &io, Vec3 &vi)
{
	char inBuf[Vec3::MAX_INPUT_STRING];
	io >> inBuf; // operator>>(ostream&, char*); -- or is it istream?
	//!FIX need to convert string to vector here
//	vi = inBuf;	// String::operator=(const char*)
	return io;
}

*/

