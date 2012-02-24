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
 * Options.cpp
 *
 *  Created on: Nov 21, 2010
 *      Author: Thomas Wiemann
 */

#include "Options.hpp"

namespace ascii_convert
{

Options::Options(int argc, char** argv) : m_descr("Supported options")
{

	// Create option descriptions

	m_descr.add_options()
		("help", "Produce help message")
		("inputFile", value< vector<string> >(), "An ASCII-File containing point cloud data.")
		("outputFile,o", value<string>()->default_value("out.txt"), "Name of the generated output file.")
	    ("xPos,x", value<int>()->default_value(0), "Position of the x-coordinates in the input data lines.")
	    ("yPos,y", value<int>()->default_value(1), "Position of the y-coordinates in the input data lines.")
	    ("zPos,z", value<int>()->default_value(2), "Position of the z-coordinates in the input data lines.")
	    ("sx", value<float>()->default_value(1.0), "Scaling factor for the x coordinates.")
	    ("sy", value<float>()->default_value(1.0), "Scaling factor for the y coordinates.")
	    ("sz", value<float>()->default_value(1.0), "Scaling factor for the z coordinates.")
	    ("rPos,r", value<int>()->default_value(-1), "Position of the red color component in the input data lines. (-1) means no color information")
	    ("gPos,g", value<int>()->default_value(-1), "Position of the green color component in the input data lines. (-1) means no color information")
	    ("bPos,b", value<int>()->default_value(-1), "Position of the blue color component in the input data lines. (-1) means no color information")
	    ("iPos,i", value<int>()->default_value(-1), "Position of the intensity information input data lines. (-1) means no intensity information")
	    ("convert,c", "Convert intensity into color")
		;

	m_pdescr.add("inputFile", -1);

	// Parse command line and generate variables map
	store(command_line_parser(argc, argv).options(m_descr).positional(m_pdescr).run(), m_variables);
	notify(m_variables);

  if(m_variables.count("help")) {
    ::std::cout<< m_descr << ::std::endl;
    exit(-1);
  }


}

bool  Options::convertRemission() const
{
    return m_variables.count("convert");
}

string Options::inputFile() const
{
    return (m_variables["inputFile"].as< vector<string> >())[0];
}

string Options::outputFile() const
{
    return (m_variables["outputFile"].as< string>());
}

bool Options::printUsage() const
{
  if(!m_variables.count("inputFile"))
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

Options::~Options() {
	// TODO Auto-generated destructor stub
}

} // namespace reconstruct
