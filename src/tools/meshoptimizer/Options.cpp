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
#include "config/lvropenmp.hpp"

namespace meshopt{

Options::Options(int argc, char** argv) : m_descr("Supported options")
{

	// Create option descriptions

	m_descr.add_options()
		        ("help", "Produce help message")
		        ("inputFile", value< vector<string> >(), "Input file name. Supported formats are ASCII (.pts, .xyz) and .ply")
			    ("optimizePlanes,o", "Shift all triangle vertices of a cluster onto their shared plane")
                ("clusterPlanes,c", "Cluster planar regions based on normal threshold, do not shift vertices into regression plane.")
		        ("cleanContours", value<int>(&m_cleanContourIterations)->default_value(0), "Remove noise artifacts from contours. Same values are between 2 and 4")
                ("planeIterations", value<int>(&m_planeIterations)->default_value(3), "Number of iterations for plane optimization")
                ("fillHoles,f", value<int>(&m_fillHoles)->default_value(30), "Maximum size for hole filling")
                ("rda", value<int>(&m_rda)->default_value(0), "Remove dangling artifacts, i.e. remove the n smallest not connected surfaces")
		        ("pnt", value<float>(&m_planeNormalThreshold)->default_value(0.85), "(Plane Normal Threshold) Normal threshold for plane optimization. Default 0.85 equals about 3 degrees.")
		        ("smallRegionThreshold", value<int>(&m_smallRegionThreshold)->default_value(0), "Threshold for small region removal. If 0 nothing will be deleted.")
     		    ("mp", value<int>(&m_minPlaneSize)->default_value(7), "Minimum value for plane optimzation")
		        ("retesselate,t", "Retesselate regions that are in a regression plane. Implies --optimizePlanes.")
		        ("lft", value<float>(&m_lineFusionThreshold)->default_value(0.01), "(Line Fusion Threshold) Threshold for fusing line segments while tesselating.")		        ("classifier", value<string>(&m_classifier)->default_value("PlaneSimpsons"),"Classfier object used to color the mesh.")
		        ("depth", value<int>(&m_depth)->default_value(100), "Maximum recursion depth for region growing.")
		        ("recalcNormals,r", "Always estimate normals, even if given in .ply file.")
		        ("threads", value<int>(&m_numThreads)->default_value( lvr::OpenMPConfig::getNumThreads() ), "Number of threads")
		        ;

	m_pdescr.add("inputFile", -1);

	// Parse command line and generate variables map
	store(command_line_parser(argc, argv).options(m_descr).positional(m_pdescr).run(), m_variables);
	notify(m_variables);

  if(m_variables.count("help")) {
    ::std::cout<< m_descr << ::std::endl;
  }

}

int Options::getNumThreads() const
{
	return m_variables["threads"].as<int>();
}


int Options::getPlaneIterations() const
{
    return m_variables["planeIterations"].as<int>();
}

string Options::getInputFileName() const
{
	return (m_variables["inputFile"].as< vector<string> >())[0];
}

string Options::getClassifier() const
{
    return (m_variables["classifier"].as< string >());
}

int    Options::getDanglingArtifacts() const
{
    return (m_variables["rda"].as<int> ());
}

int    Options::getFillHoles() const
{
    return (m_variables["fillHoles"].as<int> ());
}

int   Options::getMinPlaneSize() const
{
    return (m_variables["mp"].as<int> ());
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


bool Options::optimizePlanes() const
{
	return m_variables.count("optimizePlanes")
        || m_variables.count("retesselate");
}

bool Options::clusterPlanes() const
{
	return m_variables.count("clusterPlanes");
}

bool  Options::colorRegions() const
{
    return m_variables.count("colorRegions");
}

bool Options::retesselate() const
{
    return m_variables.count("retesselate");
}


float Options::getNormalThreshold() const
{
    return m_variables["pnt"].as<float>();
}

int   Options::getSmallRegionThreshold() const
{
    return m_variables["smallRegionThreshold"].as<int>();
}

int   Options::getCleanContourIterations() const
{
	return m_variables["cleanContours"].as<int>();
}


int Options::getDepth() const
{
	return m_depth;
}

float Options::getLineFusionThreshold() const
{
    return m_variables["lft"].as<float>();
}

Options::~Options() {
	// TODO Auto-generated destructor stub
}

} // namespace meshopt

