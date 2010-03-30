/*
 * PLYELement.cpp
 *
 *  Created on: 25.11.2009
 *      Author: twiemann
 */

#include "PLYElement.h"
#include "PLYProperty.h"


PLYElement::PLYElement()
{

}

PLYElement::PLYElement(string name, size_t count)
{
	m_name = name;
	m_count = count;
}


PLYElement::~PLYElement() {
	// TODO Auto-generated destructor stub
}

void PLYElement::addProperty(string name, string value_type, string count_type)
{
	bool list_property = (count_type != "");

	// First Create a scalar property
	Property* s = new ScalarProperty(name, value_type);


	if(list_property)
	{
		Property* l = new ListProperty(s, count_type);
		m_properties.push_back(l);
	}
	else
	{
		m_properties.push_back(s);
	}

}

size_t PLYElement::getCount()
{
	return m_count;
}

string PLYElement::getName()
{
	return m_name;
}

vector<Property*>::iterator PLYElement::getFirstProperty()
{
	return m_properties.begin();
}

vector<Property*>::iterator PLYElement::getLastProperty()
{
	return m_properties.end();
}

void PLYElement::printProperties(ofstream &str)
{
	for(size_t i = 0; i < m_properties.size(); i++)
	{
		Property *p = m_properties[i];
		if(p->isList())
		{
			str << "property list "
				<< p->getCountTypeStr() 	<< " "
				<< p->getElementTypeStr()	<< " "
				<< p->getName()				<< endl;
		}
		else
		{
			str << "property "
				<< p->getElementTypeStr() 	<< " "
				<< p->getName()				<< endl;
		}
	}
}


