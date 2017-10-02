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

namespace registration
{

Options::Options(int argc, char** argv) : m_descr("Supported options")
{

	// Create option descriptions

	m_descr.add_options()
	("help", "Produce help message")
	("maxIterations", value<int>(&m_maxIterations)->default_value( 50 ), "Maximum number of ICP iterations")
	("maxDistance", value<double>(&m_maxDistance)->default_value( 25.0 ), "Maximum squared distance in point pair generation")
	("rx", value<double>(&m_rx)->default_value( 0.0 ), "Estimated rotation around the x axis")
	("ry", value<double>(&m_ry)->default_value( 0.0 ), "Estimated rotation around the y axis")
	("rz", value<double>(&m_rz)->default_value( 0.0 ), "Estimated rotation around the z axis")
    ("tx", value<double>(&m_tx)->default_value( 0.0 ), "Estimated translation in x direction")
    ("ty", value<double>(&m_ty)->default_value( 0.0 ), "Estimated translation in y direction")
    ("tz", value<double>(&m_tz)->default_value( 0.0 ), "Estimated translation in z direction")
	("epsilon", value<double>(&m_epsilon)->default_value( 0.00001 ), "Minimum change between two ICP steps that is needed to proceed (i.e. convergence criterion)")
    ("dataCloud", value<string>(&m_dataName)->default_value("data.ply"), "Reference point cloud")
    ("modelCloud", value<string>(&m_modelName)->default_value("model.ply"), "Model point cloud")
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



Options::~Options()
{
	// TODO Auto-generated destructor stub
}

}

