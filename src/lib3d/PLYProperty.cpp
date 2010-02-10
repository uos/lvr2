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

