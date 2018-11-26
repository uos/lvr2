/**
 * Copyright (c) 2018, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
