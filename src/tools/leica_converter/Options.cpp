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

namespace leica_convert
{

Options::Options(int argc, char** argv) : m_descr("Supported options")
{

	// Create option descriptions

	m_descr.add_options()
		("help", "Produce help message")
		("in", value<string>()->default_value("./"), "A directory containing scan data for conversion.")
		("out", value<string>()->default_value("./"), "The target directory for converted data.")
		("outputFormat", value<string>()->default_value("SLAM"), "Type of target data. Choose from SLAM, MERGE, SLAM_COLOR, PLY, 3D.")
		("inputFormat", value<string>()->default_value("ALL"), "Type of parsed input data. Choose from ALL, 3D, PLY, DAT, LAS.")
	    ("filter", value<bool>()->default_value(false), "Filter input data.")
	    ("k", value<int>()->default_value(1), "k neighborhood for filtering.")
	    ("sigma", value<float>()->default_value(1.0), "Deviation for outlier filter.")
	    ("targetSize", value<int>()->default_value(100000), "Target size (reduction) for the iput scans.")
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

string 	Options::getInputDir() const
{
	return m_variables["in"].as<string>();
}

string 	Options::getOutputDir() const
{
	return m_variables["out"].as<string>();
}

string 	Options::getOutputFormat() const
{
	return m_variables["outputFormat"].as<string>();
}


string 	Options::getInputFormat() const
{
	return m_variables["inputFormat"].as<string>();
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
