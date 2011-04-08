#ifndef __PLY_TRAITS_HPP__
#define __PLY_TRAITS_HPP__

namespace lssr
{

#include <string>
using std::string;

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;

template<typename T>
struct PlyTraits
{
	PlyTraits()
	{
		SIZE = 0;
		NAME = "unknown";
	}
	static size_t SIZE;
	static string NAME;
};

template< >
struct PlyTraits<char>
{
	PlyTraits()
    {
		SIZE = 1;
		NAME = "char";
    }
	static size_t SIZE;
	static string NAME;
};

template< >
struct PlyTraits<uchar>
{
	PlyTraits()
    {
		SIZE = 1;
		NAME = "uchar";
    }
	static size_t SIZE;
	static string NAME;
};

template< >
struct PlyTraits<short>
{
	PlyTraits()
    {
		SIZE = 2;
		NAME = "short";
    }
	static size_t SIZE;
	static string NAME;
};

template< >
struct PlyTraits<ushort>
{
	PlyTraits()
    {
		SIZE = 2;
		NAME = "ushort";
    }
	static size_t SIZE;
	static string NAME;
};

template< >
struct PlyTraits<int>
{
	PlyTraits()
    {
		SIZE = 4;
		NAME = "int";
    }
	static size_t SIZE;
	static string NAME;
};

template< >
struct PlyTraits<uint>
{
	PlyTraits()
    {
		SIZE = 4;
		NAME = "uint";
    }
	static size_t SIZE;
	static string NAME;
};

template< >
struct PlyTraits<float>
{
	PlyTraits()
    {
		SIZE = 4;
		NAME = "float";
    }
	static size_t SIZE;
	static string NAME;
};

template< >
struct PlyTraits<double>
{
	PlyTraits()
    {
		SIZE = 8;
		NAME = "float";
    }
	static size_t SIZE;
	static string NAME;
};




} // namespace lssr





#endif
