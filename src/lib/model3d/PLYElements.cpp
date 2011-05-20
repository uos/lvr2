#include "PLYElements.h"
#include "PLYProperty.h"

////////////////////////////////////////////////////////////////////////////////
// ------------------------- PROPERTY DESCRIPTION ------------------------------
////////////////////////////////////////////////////////////////////////////////

ostream& operator<<(ostream& os, const PropertyDescription &p)
{
	if(p.isScalar())
	{
		os << "property" << " " << p.getValueTypeStr() << " " << p.getNameStr() << endl;
	}
	else
	{
		os << "property" 		    << " "
		   << "list"     		    << " "
		   << p.getCountTypeStr() 	<< " "
		   << p.getValueTypeStr() 	<< " "
		   << p.getNameStr() 		<< endl;
	}
	return os;
}

ostream& operator<<(ostream& os, const ElementDescription &e)
{
	os << "element" << " " << e.getStrName() << " " << e.getCount() << endl;
	e.printProperties(os);
	return os;
}

PropertyDescription::PropertyDescription()
{

	m_name			= property_unknown;
	m_countType		= type_unknown;
	m_valueType		= type_unknown;
	m_scalar		= true;
}

PropertyDescription::PropertyDescription(const PropertyDescription &other)
{
	m_name			= other.m_name;
	m_countType 	= other.m_countType;
	m_valueType		= other.m_valueType;
	m_scalar		= other.m_scalar;
}

PropertyDescription::PropertyDescription(property_name n, type_name value_t, type_name count_t)
{
	m_name = n;
	m_valueType = value_t;
	m_countType = count_t;
	m_scalar = (m_name != property_vertex_list);
 }

property_name PropertyDescription::getName() const
{
	return m_name;
}
type_name PropertyDescription::getCountType() const
{
	return m_countType;
}

type_name PropertyDescription::getValueType() const
{
	return m_valueType;
}

bool PropertyDescription::isScalar() const
{
	return m_scalar;
}

PropertyDescription::PropertyDescription(string description)
{

	// Transoform string to low characters
	std::transform(description.begin(), description.end(), description.begin(), lower());

	// Initialize with undefined values a
	m_name			= property_unknown;
	m_countType		= type_unknown;
	m_valueType		= type_unknown;
	m_scalar		= true;

	if(countBlanks(description) == 1)
	{

		// Description format: "<typename> <identifier>". Search for
		// the delimiting ' '-character.
		size_t pos = description.find(" ", 0);

		// Get substring of property type
		string s_typename = description.substr(0, pos);

		// Get substring of property identifier
		string s_identyfier = description.substr(pos + 1);

		// Check if data type and property name are supported
		m_name = stringToProperty(s_identyfier);

		if(m_name == property_unknown)
		{
			cout << "PropertyDescription: Unsupported property: '"
			     << s_identyfier << "'." << endl;
		}

		m_valueType = stringToType(s_typename);

		if(m_valueType == type_unknown)
		{
			cout << "PropertyDescription: Invalid value type: '"
				 << s_typename << "'." << endl;
		}

	} else if(countBlanks(description) == 3) {

		m_scalar = false;

		// Description format: "list <typename count> <typename element> <identifier>.
		// In this case we have to search for the two delimiting blanks
		size_t pos1 = description.find(" ", 0);
		size_t pos2 = description.find(" ", pos1 + 2);
		size_t pos3 = description.find(" ", pos2 + 2);

		// Get substrings
		string s_list 		= description.substr(0, pos1);
		string s_counttype 	= description.substr(pos1 + 1, pos2 - pos1 - 1);
		string s_valuetype  = description.substr(pos2 + 1, pos3 - pos2 - 1);
		string s_identyfier = description.substr(pos3 + 1);

		// Check list string
		if(s_list != "list")
		{
			cout << "PropertyDescription: List property doesn't start with keyword 'list': '"
				 << s_list << "'." << endl;
			return;
		}

		// Check other substrings and assign values if ok
		m_name = stringToProperty(s_identyfier);
		if(m_name == property_unknown)
		{
			cout << "PropertyDescription: Unsupported property: '"
				 << s_identyfier << "'." << endl;
		}

		m_countType = stringToType(s_counttype);
		if(m_countType == type_unknown)
		{
			cout << "PropertyDescription: Invalid count type: '"
				 << s_counttype << "'." << endl;
		}

		m_valueType = stringToType(s_valuetype);
		if(m_valueType == type_unknown)
		{
			cout << "PropertyDescription: Invalid value type: '"
				 << s_valuetype << "'." << endl;
		}


	} else {

		// If we got into this case we have encountered an invalid string
		cout << "PropertyDescription: Invalid property description string: '"
			 << description << "'." << endl;

	}

}

int PropertyDescription::countBlanks(string s) {

	// Simply iterate through string and count the
	// number of blanks.

	//TODO: Check for multiple blanks while parsing
	int count = 0;
	for(size_t i = 0; i < s.length(); i++)
	{
		if(s[i] == ' ')
		{
			count++;
		}
	}
	return count;
}

type_name PropertyDescription::stringToType(string s)
{
	// Check for known types
	if(s == "char") 			return type_char;
	if(s == "uchar")			return type_uchar;
	if(s == "short")			return type_short;
	if(s == "ushort")			return type_ushort;
	if(s == "int")				return type_int;
	if(s == "uint")				return type_uint;
	if(s == "float")			return type_float;
	if(s == "double")   		return type_double;

	// If name is unknown return default value
	return type_unknown;

}

size_t PropertyDescription::getTypeLength(type_name type) const
{
	// Determine size of property element (here we use a switch
	// to explicitly determine the byte size of each used data type.
	switch(type)
	{
	case(type_uchar): 			return sizeof(unsigned char);
	case(type_char):			return sizeof(char);
	case(type_short):			return sizeof(short);
	case(type_ushort):			return sizeof(unsigned short);
	case(type_uint):			return sizeof(unsigned int);
	case(type_int):				return sizeof(int);
	case(type_float):			return sizeof(float);
	case(type_double):			return sizeof(double);
	case(type_unknown): 		break;
	}
	return 0;
}

property_name PropertyDescription::stringToProperty(string s)
{
	// Check for known types
	if (s == "x") 				return property_x;
	if (s == "y") 				return property_y;
	if (s == "z") 				return property_z;
	if (s == "r") 				return property_r;
	if (s == "g") 				return property_g;
	if (s == "b") 				return property_b;
	if (s == "vertex_list")		return property_vertex_list;

	// If name is unknown return default value
	return property_unknown;
}

string PropertyDescription::valueToStr(type_name n) const
{
	switch(n)
	{
	case type_char: 			return "char";
	case type_uchar:			return "uchar";
	case type_short:			return "short";
	case type_ushort:			return "ushort";
	case type_int:				return "int";
	case type_uint:				return "uint";
	case type_float:			return "float";
	case type_double:			return "double";
	case type_unknown:			return "unknown";
	}

	// If name is unknown return default value
	return "unknown";
}

string PropertyDescription::getNameStr() const
{
	switch(m_name)
	{
	case property_x: 			return "x";
	case property_y:			return "y";
	case property_z:			return "z";
	case property_r:			return "r";
	case property_g:			return "g";
	case property_b:			return "b";
	case property_vertex_list:	return "vertex_list";
	case property_unknown:		return "unknown";
	}

	// If name is unknown return default value
	return "unknown";
}

string PropertyDescription::getCountTypeStr() const
{
	return valueToStr(m_countType);
}

string PropertyDescription::getValueTypeStr() const
{
	return valueToStr(m_valueType);
}

///////////////////////////////////////////////////////////////////////////////
// -------------------------- ELEMENT DESCRIPTION -----------------------------
///////////////////////////////////////////////////////////////////////////////

ElementDescription::ElementDescription()
{
	m_name			= element_unknown;
	m_count			= 0;
	m_properties.clear();
}

element_name ElementDescription::getName() const
{
	return m_name;
}


size_t ElementDescription::getCount() const
{
	return m_count;
}

void ElementDescription::printProperties(ostream &os) const
{
	// Here we have to use []-acces because of the ***ing const
	for(size_t i = 0; i < m_properties.size(); i++)
	{
		PropertyDescription p = m_properties[i];
		if(p.isScalar())
		{
			os << "property"            << " "
			   << p.getValueTypeStr()   << " "
			   << p.getNameStr()        << endl;
		}
		else
		{
			os << "property" 			<< " "
			   << "list"     			<< " "
			   << p.getCountTypeStr() 	<< " "
			   << p.getValueTypeStr() 	<< " "
			   << p.getNameStr() 		<< endl;
		}
	}
}

ElementDescription::ElementDescription(string description)
{

	// Initialize with standard values
	m_name = element_unknown;
	m_count = 0;

	// Convert string to lower case
	std::transform(description.begin(), description.end(),
			       description.begin(), lower());

	// Parse string. Valid format is "<name> <count>"
	size_t pos = description.find(" ", 0);
	string s_name 	= description.substr(0, pos);
	string s_count	= description.substr(pos + 1, description.length() - pos - 1);

	// Check for known element name
	setName(s_name);

	if(m_name == element_unknown)
	{
		cout << "ElementDescription: Unsupported element: '"
			 << s_name << "'." << endl;
		return;
	}

	// Convert cont value to size_t
	stringstream ss(s_count);
	ss >> m_count;
}

ElementDescription::ElementDescription(element_name n, size_t count)
{
	m_name 			= n;
	m_count			= count;
}

ElementDescription::ElementDescription(const ElementDescription &other)
{
	m_name 			= other.m_name;
	m_count			= other.m_count;
	m_properties 	= other.m_properties;
}


void ElementDescription::addProperty(PropertyDescription p)
{
	m_properties.push_back(p);
}

size_t ElementDescription::getPropertyCount() const
{
	return m_properties.size();
}

PropertyDescription ElementDescription::getProperty(int index) 	const
{
	if((size_t)index < m_properties.size())
	{
		return m_properties[index];
	}
	else
	{
		return PropertyDescription();
	}
}

void ElementDescription::setName(string s)
{
	if(s == "vertex") {
		m_name = element_vertex;
	}
	else if(s == "face")
	{
		m_name = element_face;
	}
	else
	{
		m_name = element_unknown;
	}
}

string ElementDescription::getStrName() const
{
	switch(m_name)
	{
	case element_vertex : 	return "vertex";
	case element_face:		return "face";
	case element_unknown:	return "unknown";
	}

	// If something went horribly wrong, we can still
	// return "unknown"....
	return "unknown";
}

