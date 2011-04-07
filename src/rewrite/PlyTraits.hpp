#ifndef __PLY_TRAITS_HPP__
#define __PLY_TRAITS_HPP__

#include <string>
using std::string;

enum plyType { CHAR, UCHAR, SHORT, USHORT, INT, UINT, FLOAT, DOUBLE, INV};

template<typename T>
struct PlyTraits
{
	static const size_t SIZE = 0;
	static const string NAME = string("NAME");
};





#endif
