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
	Property* s;

	// In every case we have to generate a scalar property
	if(value_type == "uchar")
	{
		s = new ScalarProperty<unsigned char>(name, value_type);
	}
	else if(value_type == "char")
	{
		s = new ScalarProperty<char>(name, value_type);
	}
	else if(value_type == "short")
	{
		s = new ScalarProperty<short>(name, value_type);
	}
	else if(value_type == "ushort")
	{
		s = new ScalarProperty<unsigned short>(name, value_type);
	}
	else if(value_type == "int")
	{
		s = new ScalarProperty<int>(name, value_type);
	}
	else if(value_type == "uint")
	{
		s = new ScalarProperty<unsigned int>(name, value_type);
	}
	else if(value_type == "float")
	{
		s = new ScalarProperty<float>(name, value_type);
	}
	else if(value_type == "double")
	{
		s = new ScalarProperty<double>(name, value_type);
	}
	else
	{
		cout << "PLYElement: Unsupported property data type: " << value_type << endl;
	}

	if(list_property)
	{
		Property* l;
		if(value_type == "uchar")
		{
			l = new ListProperty<unsigned char>(s, count_type);
		}
		else if(value_type == "char")
		{
			l = new ListProperty<char>(s, count_type);
		}
		else if(value_type == "short")
		{
			l = new ListProperty<short>(s, count_type);
		}
		else if(value_type == "ushort")
		{
			l = new ListProperty<unsigned short>(s, count_type);
		}
		else if(value_type == "int")
		{
			l = new ListProperty<int>(s, count_type);
		}
		else if(value_type == "uint")
		{
			l = new ListProperty<unsigned int>(s, count_type);
		}
		else
		{
			cout << "PLYElement: Unsupported list property data type: " << value_type << endl;
		}
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


