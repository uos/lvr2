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



#include "Options.hpp"
#include <omp.h>

namespace normals
{

Options::Options(int argc, char** argv) : m_descr("Supported options")
{

	// Create option descriptions

	m_descr.add_options()
	("help", "Produce help message")
	("targetSize,t", value<int>(&m_targetSize)->default_value( -1 ), "Number of desired points in the outpur file. Negative values indicate no reduction.")
	("inputDirectory,i", value<string>(&m_inputDirectory)->default_value( "./" ), "Directory containing scans with normals.")
	("outputFile,o", value<string>(&m_outputFile)->default_value( "normals.ply" ), "Output file. Supported are .ply and .3d.")
	("start,s", value<int>(&m_start)->default_value( -1 ), "First scan to convert. Set to -1 for auto detection.")
	("end,e", value<int>(&m_end)->default_value( -1 ), "Last scan to scanvert. Set to -1 for auto dection.")
    ("ki", value<int>(&m_ki)->default_value( -1 ), "Number of nearest neighbors for normal interpolation.")
	;

	m_pdescr.add("inputDirectory", -1);

	// Parse command line and generate variables map
	store(command_line_parser(argc, argv).options(m_descr).positional(m_pdescr).run(), m_variables);
	notify(m_variables);

	if(m_variables.count("help"))
	{
		::std::cout << m_descr << ::std::endl;
	}

}



Options::~Options()
{
	// TODO Auto-generated destructor stub
}

}

