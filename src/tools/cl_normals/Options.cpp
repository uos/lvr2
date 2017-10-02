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

namespace cl_normals
{

Options::Options(int argc, char** argv) : m_descr("Supported options")
{

	// Create option descriptions

	m_descr.add_options()
	("help", "Produce help message")
    ("inputFile", value< vector<string> >(), "Input file name. ")
    ("outputFile,o",    value<string>(&m_outputFile)->default_value("normals.ply"), "Output file name.")
    ("ransac", "Set this flag for RANSAC based normal estimation.")
    ("pca", "Set this flag for RANSAC based normal estimation.")
    ("kn", value<int>(&m_kn)->default_value(10), "Number of normals used for normal estimation")
    ("ki", value<int>(&m_ki)->default_value(10), "Number of normals used for normal interpolation")
    ("kd", value<int>(&m_kd)->default_value(5), "Number of normals used for distance calculation")
    ("flipx", value<float>(&m_flipx)->default_value(100000.0), "Flippoint x" )
    ("flipy", value<float>(&m_flipy)->default_value(100000.0), "Flippoint y" )
    ("flipz", value<float>(&m_flipz)->default_value(100000.0), "Flippoint z" )
    ("reconstruct,r","Reconstruct after normal calculation")
    ("exportPointNormals,e","save Pointnormals before reconstruction")
    ("voxelsize,v",value<float>(&m_voxelsize)->default_value(10.0),"voxelsize for marching cubes")
    ;

    m_pdescr.add("inputFile", -1);

	// Parse command line and generate variables map
	store(command_line_parser(argc, argv).options(m_descr).positional(m_pdescr).run(), m_variables);
	notify(m_variables);

	if(m_variables.count("help"))
	{
		::std::cout << m_descr << ::std::endl;
        exit(-1);
	}

}



Options::~Options()
{
	// TODO Auto-generated destructor stub
}

}

