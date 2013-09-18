/* Copyright (C) 2011 Uni Osnabrück
 * This file is part of the LAS VEGAS Reconstruction Toolkit,
 *
 * LAS VEGAS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * LAS VEGAS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA
 */

 /*
 * Options.h
 *
 *  Created on: August 21, 2013
 *      Author: Henning Deeken {hdeeken@uos.de}
 */

#include "Options.hpp"

namespace slicer{

Options::Options(int argc, char** argv) : m_descr("Supported options")
{

	// Create option descriptions

	m_descr.add_options()
		        ("help", "Produce help message")
		        ("input", value< vector<string> >(), "Input file name. Supported formats are .ply")
		        ("dimension", value< vector<string> >(), "Dimension parameter for the AABB Search.")
		        ("value", value< vector<double> >(), "Dimension value for the AABB Search.")
        ;

	m_pdescr.add("input", -1);
	m_pdescr.add("dimension", -1);
	m_pdescr.add("value", -1);

	// Parse command line and generate variables map
	store(command_line_parser(argc, argv).options(m_descr).positional(m_pdescr).run(), m_variables);
	notify(m_variables);

  if(m_variables.count("help")) {
	  
    ::std::cout<< m_descr << ::std::endl;
  }
}

Options::~Options() {
	// TODO Auto-generated destructor stub
}

string Options::getInputFileName() const
{
	return (m_variables["input"].as< vector<string> >())[0];
}

string Options::getDimension() const
{
	return (m_variables["dimension"].as< vector<string> >())[0];
}

double Options::getValue() const
{
	return (m_variables["value"].as< vector<double> >())[0];
}

bool Options::printUsage() const
{
  if(!m_variables.count("input"))
    {
      cout << "Error: You must specify an input file." << endl;
      cout << endl;
      cout << m_descr << endl;
      return true;
    }
    
    if(!m_variables.count("dimension") || m_variables.count("dimension") > 1)
    {
      cout << "Error: You must specify exactly one dimension d out of x, y, z." << endl;
      cout << endl;
      cout << m_descr << endl;
      return true;
    }
    
     if(!m_variables.count("value") || m_variables.count("value") > 1)
    {
      cout << "Error: You must specify exactly one value." << endl;
      cout << endl;
      cout << m_descr << endl;
      return true;
    }
    
  if(m_variables.count("help"))
    {
      cout << endl;
      cout << m_descr << endl;
      return true;
    }
  return false;
}

} // namespace slicer

