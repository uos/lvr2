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
 *  Created on: July 14, 2013
 *      Author: Henning Deeken {hdeeken@uos.de}
 *              Ann-Katrin Häuser {ahaeuser@uos.de}
 */

#include "Options.hpp"

namespace fusion{

Options::Options(int argc, char** argv) : m_descr("Supported options")
{

	// Create option descriptions

	m_descr.add_options()
		        ("help", "Produce help message")
		        ("mesh1", value< vector<string> >(), "Input file name for mesh1. Supported formats are .ply")
		        ("mesh2", value< vector<string> >(), "Input file name for mesh2. Supported formats are .ply")
		        ("fusion", value< vector<string> >(), "Input file name for mesh2. Supported formats are .ply")
		        ("t", value< vector<double> >(), "Distance treshold for AABB Search.")
				("v", "Verbosity On.")
        ;
		;

	m_pdescr.add("mesh1", -1);
	m_pdescr.add("mesh2", -1);

	// Parse command line and generate variables map
	store(command_line_parser(argc, argv).options(m_descr).positional(m_pdescr).run(), m_variables);
	notify(m_variables);

  if(m_variables.count("help")) 
  {
	  
    ::std::cout<< m_descr << ::std::endl;
  }
}

Options::~Options() {
	// TODO Auto-generated destructor stub
}

bool Options::getVerbosity() const
{
	return (m_variables.count("v"));
}

string Options::getMesh1FileName() const
{
	return (m_variables["mesh1"].as< vector<string> >())[0];
}

string Options::getMesh2FileName() const
{
	return (m_variables["mesh2"].as< vector<string> >())[0];
}

string Options::getFusionMeshFileName() const
{
	return (m_variables["fusion"].as< vector<string> >())[0];
}

double Options::getDistanceTreshold() const
{
	return (m_variables["t"].as< vector<double> >())[0];
}

bool Options::printUsage() const
{
  if(!m_variables.count("mesh1") || !m_variables.count("mesh2"))
    {
      cout << "Error: You must specify an input file." << endl;
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

bool Options::outputFileNameSet() const
{
	return (m_variables["fusion"].as< vector<string> >()).size() > 0;
}

} // namespace fusion

