/*
 * PLYProperty.cpp
 *
 *  Created on: 25.11.2009
 *      Author: twiemann
 */

#include "PLYProperty.h"

///////////////////////////////////////////////////////////////////////////////
// ----------------------------- Base Property --------------------------------
///////////////////////////////////////////////////////////////////////////////

Property::Property()
{
	m_name = "";
	m_elementTypeName = "";
	m_countTypeName = "";
	m_size = 0;
}

Property::~Property()
{
	// Nothing to do.
}

string Property::getName()
{
	return m_name;
}

bool Property::isList()
{
	return (getCountSize() != 0);
}

size_t Property::calcSize(string& s)
{
	if(s == "char" || s == "uchar")
	{
		return  1;
	}
	else if(s == "short" || s == "ushort")
	{
		return  2;
	}
	else if(s == "int" || s == "uint" || s == "float")
	{
		return  4;
	}
	else if(s == "double")
	{
		return  8;
	}
	else
	{
		cout << "Property::calcSize(): Usupported element type " << m_elementTypeName << "." << endl;
	}
}

///////////////////////////////////////////////////////////////////////////////
// ----------------------------- Scalar Property ------------------------------
///////////////////////////////////////////////////////////////////////////////


ScalarProperty::ScalarProperty(string name, string type_dscr)
{
	m_name = name;
	m_countTypeName = "";
	m_elementTypeName = type_dscr;
	m_size = calcSize(type_dscr);
}


ScalarProperty::~ScalarProperty()
{

}


size_t ScalarProperty::getCountSize()
{
	return 0;
}


size_t ScalarProperty::getValueSize()
{
	return m_size;
}


string ScalarProperty::getElementTypeStr()
{
	return m_elementTypeName;
}


string ScalarProperty::getCountTypeStr()
{
	return "";
}

///////////////////////////////////////////////////////////////////////////////
// ----------------------------- List Property --------------------------------
///////////////////////////////////////////////////////////////////////////////


ListProperty::ListProperty(Property* p, string count_type_dscr)
{
	// TO DO: Writer a proper Property constructor to avoid manual assignment
	m_property = p;
	m_countTypeName = count_type_dscr;
	m_name = p->getName();
	m_size = calcSize(count_type_dscr);
}

ListProperty::~ListProperty()
{
	//TODO: delete m_property;
}

size_t ListProperty::getCountSize()
{
	return m_size;
}


size_t ListProperty::getValueSize()
{
	return m_property->getValueSize();
}

string ListProperty::getCountTypeStr()
{
	return m_countTypeName;
}


string ListProperty::getElementTypeStr()
{
	return m_property->getElementTypeStr();
}


