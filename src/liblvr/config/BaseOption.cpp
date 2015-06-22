/*
 * BaseOption.cpp
 *
 *  Created on: Feb 4, 2015
 *      Author: twiemann
 */

#include "config/BaseOption.hpp"

namespace lvr
{

BaseOption::BaseOption(int argc, char** argv)
	: m_descr("Supported options"), m_argc(argc), m_argv(argv)
{
	m_descr.add_options()
    		("xPos,x", value<int>()->default_value(0), "Position of the x-coordinates in the input point data (according to screen coordinates).")
			("yPos,y", value<int>()->default_value(1), "Position of the y-coordinates in the input data lines (according to screen coordinates).")
			("zPos,z", value<int>()->default_value(2), "Position of the z-coordinates in the input data lines (according to screen coordinates).")
			("sx", value<float>()->default_value(1.0), "Scaling factor for the x coordinates.")
			("sy", value<float>()->default_value(1.0), "Scaling factor for the y coordinates.")
			("sz", value<float>()->default_value(1.0), "Scaling factor for the z coordinates.")
	;

}

void BaseOption::printTransformation(std::ostream& out) const
{
	out << "##### Program options: " << std::endl;
	if(m_coordinateTransform.convert)
	{
		out << "##### Transform input data\t: YES" << std::endl;
		out << "##### Position of x coordinates\t: " << x() << std::endl;
		out << "##### Position of y coordinates\t: " << y() << std::endl;
		out << "##### Position of z coordinates\t: " << z() << std::endl;
		out << "##### X-Scale\t\t\t: " << sx() << std::endl;
		out << "##### Y-Scale\t\t\t: " << sy() << std::endl;
		out << "##### Z-Scale\t\t\t: " << sz() << std::endl;
	}
	else
	{
		out << "##### Transform input data\t: NO" << std::endl;
	}
}

void BaseOption::setupInputTransformation()
{

}

void BaseOption::setup()
{
	m_pdescr.add("inputFile", -1);

	// Parse command line and generate variables map
	store(command_line_parser(m_argc, m_argv).options(m_descr).positional(m_pdescr).run(), m_variables);
	notify(m_variables);

	if(m_variables.count("help")) {
		::std::cout<< m_descr << ::std::endl;
	}

	if(sx() != 1.0 || sy() != 1.0 || sz() != 0 || x() != 1 || y() != 1 || z() != 1)
	{
		m_coordinateTransform.convert = true;
		m_coordinateTransform.x = x();
		m_coordinateTransform.y = y();
		m_coordinateTransform.z = z();
		m_coordinateTransform.sx = sx();
		m_coordinateTransform.sy = sy();
		m_coordinateTransform.sz = sz();

		ModelFactory::m_transform = m_coordinateTransform;
	}
}

BaseOption::~BaseOption()
{
	// Nothing to do...
}

} /* namespace lvr */
