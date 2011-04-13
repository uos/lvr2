#ifndef __PLYPROPERTY_HPP__
#define __PLYPROPERTY_HPP__

#include <string>
#include <iostream>
using std::string;

#include "PLYTraits.hpp"

namespace lssr
{

/**
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

	/// The property's name
	string 			m_name;

	/// The number of elements in the list (only for list props.)
	size_t			m_count;
};


/**
 * @brief Representation of a scalar ply property.
 */
template<typename ElementT>
class ScalarProperty : public Property
{
public:

	/**
	 * @brief Construcs a ScalarProperty with the given name
	 * @param name			The name of the property
	 */
	ScalarProperty(string name)
	{
		this->m_name = name;
		this->m_count = 0;
	}

	/**
	 * @brief Returns a string representation of the property's type
	 */
	string getElementTypeStr() { return PLYTraits<ElementT>::NAME;}

	/**
	 * @brief Returns the size of the property in bytes.
	 */
	size_t getValueSize() {return PLYTraits<ElementT>::SIZE;}

	/**
	 * @brief Always zero for scalar properties
	 */
	size_t getCountSize() { return 0;}

	/**
	 * @brief Always empty for scalar properties
	 */
	string getCountTypeStr() { return "";}

	/**
	 * @brief Always false for scalar properties
	 */
	bool isList() { return false;}
};


/**
 * @brief Representation of a list property in ply files
 */
template<typename ListT>
class ListProperty : public Property
{
public:

	/**
	 * @brief Constructor.
	 *
	 * @param count		Number of elements in the list
	 * @param p			The scalar property that is represented in the list
	 *
	 * Property lists encode repeated appearance of a special property.
	 * The reduce the number of template parameters, we use the information
	 * stored in a scalar property and store them internally. This way the
	 * list property only depends on a single template parameter (for the
	 * list type) which eases instantiation.
	 */
	ListProperty(size_t count, Property* p) : m_property(p)
	{
		this->m_name = p->getName();
		this->m_count = count;
	}

	/**
	 * @brief Returns the size of the list elements
	 */
	size_t getValueSize() { return m_property->getValueSize();}

	/**
	 * @brief Returns a string representation of the type of the
	 * 		  list elements
	 */
	string getElementTypeStr() { return m_property->getElementTypeStr();}

	/**
	 * @brief Returns the size in bytes of the counting data type
	 */
	size_t getCountSize() { return PLYTraits<ListT>::SIZE;}

	/**
	 * @brief Returns a string representation of the count data type
	 */
	string getCountTypeStr() { return PLYTraits<ListT>::NAME; }

	/**
	 * @brief Always true for list properties.
	 */
	bool isList() { return true;}

private:

	/// A scalar property for the list elements
	Property* m_property;
};



} // namespace lssr

#endif
