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

namespace filter
{

Options::Options(int argc, char** argv) : m_descr("Supported options")
{

	// Create option descriptions

	m_descr.add_options()
		("help", "Produce help message")
		("inputFile", value< vector<string> >(), "An input in a supported file formats")
		("outputFile,o", value<string>()->default_value("out.txt"), "Name of the generated output file.")
		("removeOutliers,r", "Remove outliers from data set.")
	    ("mlsDistance,m", value<float>()->default_value(0), "Max distance for MLS reconstruction.")
	    ("sorThresh,t", value<float>()->default_value(1.0), "Std. deviation threshold for outlier removal.")
	    ("sorMeank,k", value<int>()->default_value(50), "k value mean calculation for outlier removal.")
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


int Options::sorMeanK() const
{
    return (m_variables["sorMeank"].as<int>());
}


float Options::sorDevThreshold() const
{
    return (m_variables["sorThresh"].as< float>());
}


bool  Options::removeOutliers() const
{
    return (m_variables.count("removeOutliers"));
}


string Options::inputFile() const
{
    return (m_variables["inputFile"].as< vector<string> >())[0];
}

string Options::outputFile() const
{
    return (m_variables["outputFile"].as< string>());
}

float Options::mlsDistance() const
{
    return (m_variables["mlsDistance"].as< float>());
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
