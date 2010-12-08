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
	size_t  calcSize(string& s);

};

class ScalarProperty : public Property{

public:
	ScalarProperty(string Name, string type_dscr);
	~ScalarProperty();

	virtual size_t getCountSize();
	virtual size_t getValueSize();

	string getCountTypeStr();
	string getElementTypeStr();

};

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
	Property* 	m_property;
};



#endif /* PLYPROPERTY_H_ */
