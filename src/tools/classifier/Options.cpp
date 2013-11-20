/**
 * Copyright (C) 2013 Universität Osnabrück
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


/**
* @file Options.cpp
* @author Simon Herkenhoff <sherkenh@uni-osnabrueck.de>
*/

#include "Options.hpp"
#include <omp.h>
#include <fstream>

namespace classifier
{

Options::Options(int argc, char** argv) : m_descr("Supported options")
{

	// Create option descriptions

	m_descr.add_options()
	("help", "Produce help message")
	("threads", value<int>(&m_numThreads)->default_value( omp_get_num_procs() ), "Number of threads")
	("inputFile", value< vector<string> >(), "Input file name. Supported formats are ASCII (.pts, .xyz) and .ply")
	;

	m_pdescr.add("inputFile", -1);

	// Parse command line and generate variables map
	store(command_line_parser(argc, argv).options(m_descr).positional(m_pdescr).run(), m_variables);
	notify(m_variables);

	if(m_variables.count("help"))
	{
		::std::cout << m_descr << ::std::endl;
	}

}

string Options::getInputFileName() const
{
	return (m_variables["inputFile"].as< vector<string> >())[0];
}

int Options::getNumThreads() const
{
	return m_variables["threads"].as<int>();
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

Options::~Options()
{
	// TODO Auto-generated destructor stub
}

}

