/*
 * PLYElement.hpp
 *
 *  Created on: 11.04.2011
 *      Author: Thomas Wiemann
 */

#ifndef PLYELEMENT_HPP_
#define PLYELEMENT_HPP_

#include <vector>
#include <iostream>
using std::vector;
using std::ostream;
using std::endl;

#include "PLYProperty.hpp"

namespace lssr
{

/**
 * @brief Representation of an element in a ply file.
 */
class PLYElement
{
public:

	/**
	 * @brief Default ctor.
	 */
	PLYElement() : m_name(""), m_count(0) {};

	/**
	 * @brief Creates an element description with given name and count
	 *
	 * @param name			The name element
	 * @param count			The number of elements in the file
	 */
	PLYElement(string name, size_t count) : m_name(name), m_count(count){};

	/**
	 * @brief Adds a new property with the given name into the element.
	 *
	 * @param name			The name of the new property.
	 * @param elementType	The element type
	 * @
	 */
	void addProperty(string name, string elementType, string listType = "", size_t count = 0)
	{
		// First create a scalar property
		Property* p;

		// Now we have to instantiate a appropriate property class
		if(elementType == "char")
		{
			p = new ScalarProperty<char>(name);
		}
		else if(elementType == "uchar")
		{
			p = new ScalarProperty<uchar>(name);
		}
		else if(elementType == "short")
		{
			p = new ScalarProperty<short>(name);
		}
		else if(elementType == "uint")
		{
			p = new ScalarProperty<uint>(name);
		}
		else if(elementType == "int")
		{
			p = new ScalarProperty<int>(name);
		}
		else if(elementType == "float")
		{
			p = new ScalarProperty<float>(name);
		}
		else if(elementType == "double")
		{
			p = new ScalarProperty<double>(name);
		}

		// Check if we have a list
		if(listType == "")
		{
			m_properties.push_back(p);
			return;
		}
		else
		{
			Property* l;
			if(listType == "char")
			{
				l = new ListProperty<char>(count, p);
			}
			else if(listType == "uchar")
			{
				l = new ListProperty<uchar>(count, p);
			}
			else if(listType == "short")
			{
				l = new ListProperty<short>(count, p);
			}
			else if(listType == "uint")
			{
				l = new ListProperty<uint>(count, p);
			}
			else if(listType == "int")
			{
				l = new ListProperty<int>(count, p);
			}
			else if(listType == "float")
			{
				l = new ListProperty<float>(count, p);
			}
			else if(listType == "double")
			{
				l = new ListProperty<double>(count, p);
			}
			m_properties.push_back(l);
		}

	}

	/***
	 * @brief Gets an iterator to the first property in the property list.
	 */
	vector<Property*>::iterator getFirstProperty() { return m_properties.begin(); }

	/**
	 * @brief Returns an iterator to the end of the property list.
	 */
	vector<Property*>::iterator getLastProperty() { return m_properties.end(); }

	/**
	 * @brief Returns the name of the element
	 */
	string getName() {return m_name;}

	/**
	 * @brief Returns the number of elements in the ply file.
	 */
	size_t getCount() {return m_count;}

	/**
	 * @brief Prints a ply conform description of each property
	 * 		  in the element to the given stream.
	 */
	void printProperties(ostream &out)
	{
		for(vector<Property*>::iterator it = m_properties.begin();
			it != m_properties.end();
			it ++)
		{
			Property* p = *it;
			out << "property ";
			if(p->isList())
			{
				out << "list ";
				out << p->getCountTypeStr() << " ";
			}
			out << p->getElementTypeStr() << " ";
			out << p->getName() << endl;
		}
	}

private:

	/// A list of properties of the current element
	vector<Property*> m_properties;

	/// The name of the element
	string 	m_name;

	/// The number of elements of this type in the ply file.
	size_t	m_count;


};

} // namespace lssr;

#endif /* PLYELEMENT_HPP_ */
