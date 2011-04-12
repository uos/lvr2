#ifndef __PLYPROPERTY_HPP__
#define __PLYPROPERTY_HPP__

#include <string>
#include <iostream>
using std::string;

#include "PLYTraits.hpp"

namespace lssr
{

/***
 * @brief Abstract interface class for properties of elements in
 * 		  PLY files.
 */
class Property
{
public:


	/**
	 * @brief Returns the size a list count type in bytes. If
	 * 		  the property is not a list, the function will
	 * 		  return 0.
	 */
	virtual size_t getCountSize() = 0;

	/**
	 * @brief Returns the size of the property value.
	 */
	virtual size_t getValueSize() = 0;

	/**
	 * @brief Returns a string representation of of the used
	 *        type for lists. Returns an empty string if
	 *        the element is not a list.
	 */
	virtual string getCountTypeStr() = 0;

	/**
	 * @brief Returns a string representation of the property's
	 * 		  type.
	 */
	virtual string getElementTypeStr() = 0;

	/**
	 * @brief Returns the property's name
	 */
	string getName() {return m_name;}

	/**
	 * @brief Returns the number of list entries for list properties
	 */
	size_t getCount() {return m_count;}

	/**
	 * @brief Returns true if the property is a list property
	 */
	virtual bool isList() = 0;


protected:
	string 			m_name;
	size_t			m_count;
};

template<typename ElementT>
class ScalarProperty : public Property
{
public:
	ScalarProperty(string name)
	{
		this->m_name = name;
		this->m_count = 0;
	}
	string getElementTypeStr() { return PLYTraits<ElementT>::NAME;}
	size_t getValueSize() {return PLYTraits<ElementT>::SIZE;}
	size_t getCountSize() { return 0;}
	string getCountTypeStr() { return "";}
	bool isList() { return false;}
};

template<typename ListT>
class ListProperty : public Property
{
public:
	ListProperty(size_t count, Property* p) : m_property(p)
	{
		this->m_name = p->getName();
		this->m_count = count;
	}
	size_t getValueSize() { return m_property->getValueSize();}
	string getElementTypeStr() { return m_property->getElementTypeStr();}
	size_t getCountSize() { return PLYTraits<ListT>::SIZE;}
	string getCountTypeStr() { return PLYTraits<ListT>::NAME; }
	bool isList() { return true;}

private:
	Property* m_property;
};



} // namespace lssr

#endif
