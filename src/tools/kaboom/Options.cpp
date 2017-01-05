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
 *  Created on: Aug 7, 2013
 *      Author: Thomas Wiemann
 */

#include "Options.hpp"

namespace kaboom
{

Options::Options(int argc, char** argv) : m_descr("Supported options")
{

	// Create option descriptions

	m_descr.add_options()
		("help", "Produce help message")
		("inputDir", value<string>()->default_value("./"), "A directory containing several scan files for batch conversion.")
		("inputFile", value<string>()->default_value(""), "A single file to convert.")
		("outputFile", value<string>()->default_value("scan.txt"), "The name of a single output file if scans are merged. If the format can be deduced frim the file extension, the specification of --outputFormat is optional.")
		("outputDir", value<string>()->default_value("./"), "The target directory for converted data.")
		("outputFormat", value<string>()->default_value(""), "Specify the output format. Possible values are ASCII, PLY, DAT, LAS. If left empty, the format is deduced from the extension of the input files.")
	    ("filter", value<bool>()->default_value(false), "Filter input data.")
	    ("k", value<int>()->default_value(1), "k neighborhood for filtering.")
	    ("sigma", value<float>()->default_value(1.0), "Deviation for outlier filter.")
	    ("targetSize", value<int>()->default_value(100000), "Target size (reduction) for the iput scans.")
	    ("xPos,x", value<int>()->default_value(0), "Position of the x-coordinates in the input data lines.")
	    ("yPos,y", value<int>()->default_value(1), "Position of the y-coordinates in the input data lines.")
	    ("zPos,z", value<int>()->default_value(2), "Position of the z-coordinates in the input data lines.")
	    ("sx", value<float>()->default_value(1.0), "Scaling factor for the x coordinates.")
	    ("sy", value<float>()->default_value(1.0), "Scaling factor for the y coordinates.")
	    ("sz", value<float>()->default_value(1.0), "Scaling factor for the z coordinates.")
	    ("rPos,r", value<int>()->default_value(-1), "Position of the red color component in the input data lines. (-1) means no color information")
	    ("gPos,g", value<int>()->default_value(-1), "Position of the green color component in the input data lines. (-1) means no color information")
	    ("bPos,b", value<int>()->default_value(-1), "Position of the blue color component in the input data lines. (-1) means no color information")
        ("start,s", value<int>()->default_value(0), "start at scan NR")
        ("end,e", value<int>()->default_value(0), "end at scan NR")
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

string Options::getOutputFile() const
{
	return m_variables["outputFile"].as<string>();
}

string Options::getInputFile() const
{
	return m_variables["inputFile"].as<string>();
}

string 	Options::getInputDir() const
{
	return m_variables["inputDir"].as<string>();
}

string 	Options::getOutputDir() const
{
	return m_variables["outputDir"].as<string>();
}

string 	Options::getOutputFormat() const
{
	return m_variables["outputFormat"].as<string>();
}

bool	Options::filter() const
{
	return m_variables["filter"].as<bool>();
}

int		Options::getK() const
{
	return m_variables["k"].as<int>();
}

float	Options::getSigma() const
{
	return m_variables["sigma"].as<float>();
}

int		Options::getTargetSize() const
{
	return m_variables["targetSize"].as<int>();
}


Options::~Options() {
	// TODO Auto-generated destructor stub
}

} // namespace reconstruct
