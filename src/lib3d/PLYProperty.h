/*
 * PLYProperty.h
 *
 *  Created on: 25.11.2009
 *      Author: twiemann
 */

#ifndef PLYPROPERTY_H_
#define PLYPROPERTY_H_

#include <string>
#include <fstream>
#include <iostream>

using std::cout;
using std::endl;
using std::string;
using std::ostream;

class Property {
public:
	Property();
	virtual ~Property();

	string 	getName();


	bool	isList();

	virtual size_t getCountSize() = 0;
	virtual size_t getValueSize() = 0;
	virtual string getCountTypeStr() = 0;
	virtual string getElementTypeStr() = 0;

protected:
	string 	m_name;
	string 	m_elementTypeName;
	string 	m_countTypeName;

	size_t  m_size;
};

template<typename V>
class ScalarProperty : public Property{

public:
	ScalarProperty(string Name, string type_dscr);
	~ScalarProperty();

	virtual size_t getCountSize();
	virtual size_t getValueSize();

	string getCountTypeStr();
	string getElementTypeStr();

private:
	V	m_valueType;

};

template<typename C>
class ListProperty : public Property
{
public:
	ListProperty(Property* p, string type_dscr);
	~ListProperty();

	virtual size_t getCountSize();
	virtual size_t getValueSize();

	virtual string getElementTypeStr();
	virtual string getCountTypeStr();

private:
	C 			m_countType;
	Property* 	m_property;
};

///////////////////////////////////////////////////////////////////////////////
// ----------------------------- Scalar Property ------------------------------
///////////////////////////////////////////////////////////////////////////////

template<typename V>
ScalarProperty<V>::ScalarProperty(string name, string type_dscr)
{
	m_name = name;
	m_valueType = 0;
	m_countTypeName = "";
	m_elementTypeName = type_dscr;
}

template<typename V>
ScalarProperty<V>::~ScalarProperty()
{

}

template<typename V>
size_t ScalarProperty<V>::getCountSize()
{
	return 0;
}

template<typename V>
size_t ScalarProperty<V>::getValueSize()
{
	return m_size;
}

template<typename V>
string ScalarProperty<V>::getElementTypeStr()
{
	return m_elementTypeName;
}

template<typename V>
string ScalarProperty<V>::getCountTypeStr()
{
	return "";
}

///////////////////////////////////////////////////////////////////////////////
// ----------------------------- List Property --------------------------------
///////////////////////////////////////////////////////////////////////////////

template<typename C>
ListProperty<C>::ListProperty(Property* p, string count_type_dscr)
{
	// TO DO: Writer a proper Property constructor to avoid manual assignment
	m_property = p;
	m_countType = 0;
	m_countTypeName = count_type_dscr;
	m_name = p->getName();
}

template<typename C>
ListProperty<C>::~ListProperty()
{
	//TODO: delete m_property;
}

template<typename C>
size_t ListProperty<C>::getCountSize()
{
	return sizeof(C);
}

template<typename C>
size_t ListProperty<C>::getValueSize()
{
	return m_property->getValueSize();
}

template<typename C>
string ListProperty<C>::getCountTypeStr()
{
	return m_countTypeName;
}

template<typename C>
string ListProperty<C>::getElementTypeStr()
{
	return m_property->getElementTypeStr();
}


#endif /* PLYPROPERTY_H_ */
