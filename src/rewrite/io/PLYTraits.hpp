#ifndef __PLY_TRAITS_HPP__
#define __PLY_TRAITS_HPP__

#include <string>
using std::string;

namespace lssr
{

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;

template<typename T>
struct PLYTraits
{
	static size_t SIZE;
	static std::string NAME;
};

template<typename T>
string PLYTraits<T>::NAME = "unsupported";

template<typename T>
size_t PLYTraits<T>::SIZE = 0;

template< >
struct PLYTraits<char>
{
	static size_t SIZE;
	static string NAME;
};

template< >
struct PLYTraits<uchar>
{
	static size_t SIZE;
	static string NAME;
};



template< >
struct PLYTraits<short>
{
	static size_t SIZE;
	static string NAME;
};



template< >
struct PLYTraits<ushort>
{
	static size_t SIZE;
	static string NAME;
};




template< >
struct PLYTraits<int>
{
	static size_t SIZE;
	static string NAME;
};



template< >
struct PLYTraits<uint>
{
	static size_t SIZE;
	static string NAME;
};



template< >
struct PLYTraits<float>
{
	static size_t SIZE;
	static string NAME;
};




template< >
struct PLYTraits<double>
{
	static size_t SIZE;
	static string NAME;
};



} // namespace lssr

#endif
