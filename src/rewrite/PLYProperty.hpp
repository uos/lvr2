/*
 * PLYProperty.h
 *
 *  Created on: 25.11.2009
 *      Author: twiemann
 */

#ifndef PLYPROPERTY_H_
#define PLYPROPERTY_H_

#include "PlyTraits.hpp"

#include <string>
#include <fstream>
#include <iostream>

using std::cout;
using std::endl;
using std::string;
using std::ostream;

namespace lssr
{

/***
 * @brief Class to a property of an element in a .ply file
 */
//template<typename ValueT, typename CountT = bool>  // bool to indicate scalar property
//class PLYProperty
//{
//	PLYProperty(string name);
//
//	size_t countSize () { return PlyTraits<CountT>::SIZE;}
//	size_t valueSize () { return PlyTraits<ValueT>::SIZE;}
//
//	static PlyTraits<CountT> countTraits;
//	static PlyTraits<CountT> valueTraits;
//
//};

class BaseProperty
{
	virtual size_t getCountSize() = 0;
	virtual size_t getValueSize() = 0;
	virtual string getCountTypeStr() = 0;
	virtual string getElementTypeStr() = 0;
};



//class PLYProperty {
//public:
//	PLYProperty();
//	virtual ~PLYProperty();
//
//	string 	getName();
//
//
//	bool	isList();
//
//	virtual size_t getCountSize() = 0;
//	virtual size_t getValueSize() = 0;
//	virtual string getCountTypeStr() = 0;
//	virtual string getElementTypeStr() = 0;
//
//protected:
//	string 	m_name;
//	string 	m_elementTypeName;
//	string 	m_countTypeName;
//
//	size_t  m_size;
//	size_t  calcSize(string& s);
//
//};
//
//class ScalarProperty : public PLYProperty{
//
//public:
//	ScalarProperty(string Name, string type_dscr);
//	~ScalarProperty();
//
//	virtual size_t getCountSize();
//	virtual size_t getValueSize();
//
//	string getCountTypeStr();
//	string getElementTypeStr();
//
//};
//
//class ListProperty : public PLYProperty
//{
//public:
//	ListProperty(PLYProperty* p, string type_dscr);
//	~ListProperty();
//
//	virtual size_t getCountSize();
//	virtual size_t getValueSize();
//
//	virtual string getElementTypeStr();
//	virtual string getCountTypeStr();
//
//private:
//	PLYProperty* 	m_property;
//};

}

#endif /* PLYPROPERTY_H_ */
