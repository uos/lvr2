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

namespace reduce{

Options::Options(int argc, char** argv) : m_descr("Supported options")
{

    // Create option descriptions
    m_descr.add_options()
        ("help", "Produce help message")
        ("reduction,r", value<int>(&m_reduction)->default_value(1), "Reduction factor, i.e. only read every n-th point.")
        ("points,p", value<unsigned int>(&m_points)->default_value(1000000),"How much points should be left, pick random points, so that new pointcloud has n points")
        ("output,o", value<string>()->default_value("reduced.ply"), "Name of the generated output file.")
        ("inputFile", value< vector<string> >(), "inputFile")
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



int Options::reduction() const
{
    return m_variables["reduction"].as<int>();
}
unsigned int Options::points() const
{
    return m_variables["points"].as<unsigned int>();
}

vector<string> Options::inputFile() const
{
    return (m_variables["inputFile"].as< vector<string> >());
}

string Options::outputFile() const
{
    return m_variables["output"].as<string>();
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
