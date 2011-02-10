#ifndef __PLY_ELEMENTS_H__
#define __PLY_ELEMENTS_H__

#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include <locale>
#include <sstream>

using std::vector;
using std::string;
using std::cout;
using std::endl;
using std::stringstream;
using std::ostream;

namespace lssr
{

struct lower {
  int operator()(int c)
  {
    return std::tolower((unsigned char)c);
  }
};


enum type_name {
	type_char,
	type_uchar,
	type_short,
	type_ushort,
	type_int,
	type_uint,
	type_float,
	type_double,
	type_unknown
};

enum element_name  {
	element_vertex,
	element_face,
	element_unknown
};

enum property_name {
	property_x,
	property_y,
	property_z,
	property_r,
	property_g,
	property_b,
	property_vertex_list,
	property_unknown
};


class PropertyDescription {

public:
	PropertyDescription();
	PropertyDescription(string desription);
	PropertyDescription(property_name n, type_name value_t, type_name count_t = type_unknown);
	PropertyDescription(const PropertyDescription &other);

	property_name 	getName() 					const;
	type_name 		getCountType() 				const;
	type_name 		getValueType() 				const;

	string 			getCountTypeStr()			const;
	string 			getValueTypeStr()			const;
	string 			getNameStr()				const;

	bool isScalar() 							const;

	size_t			getTypeLength(type_name n)	const;

private:

	string			valueToStr(type_name n)    	const;

	type_name 		stringToType(string s);
	property_name 	stringToProperty(string s);

	int  			countBlanks(string s);

	property_name	m_name;
	type_name		m_countType;
	type_name		m_valueType;
	bool			m_scalar;
};

class ElementDescription {

public:
	ElementDescription();
	ElementDescription(string description);
	ElementDescription(element_name n, size_t count);
	ElementDescription(const ElementDescription &other);

	void 	addProperty(PropertyDescription dscr);
	void    printProperties(ostream& os)		const;

	element_name 	getName() 					const;
	size_t 	getCount() 							const;
	size_t  getPropertyCount()					const;

	string 	getStrName()						const;

	PropertyDescription getProperty(int index) 	const;


private:

	void setName(string s);

	vector<PropertyDescription>	m_properties;
	element_name				m_name;
	size_t						m_count;
};

ostream& operator<<(ostream& os, const PropertyDescription &p);
ostream& operator<<(ostream& os, const ElementDescription &p);

}

#endif
