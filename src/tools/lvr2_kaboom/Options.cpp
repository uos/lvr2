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
 *  Created on: Aug 7, 2013
 *      Author: Thomas Wiemann
 */

#include "Options.hpp"

namespace kaboom
{

Options::Options(int argc, char** argv) : lvr2::BaseOption(argc, argv), m_descr("Supported options")
{

	// Create option descriptions

	m_descr.add_options()
		("help", "Produce help message")
		("inputDir", value<string>()->default_value("./"), "A directory containing several scan files for batch conversion.")
		("inputFile", value<string>()->default_value(""), "A single file to convert.")
		("outputFile", value<string>()->default_value(""), "The name of a single output file if scans are merged. If the format can be deduced frim the file extension, the specification of --outputFormat is optional.")
		("outputDir", value<string>()->default_value("./"), "The target directory for converted data.")
		("outputFormat", value<string>()->default_value(""), "Specify the output format. Possible values are ASCII, PLY, DAT, LAS. If left empty, the format is deduced from the extension of the input files.")
	    ("filter", value<bool>()->default_value(false), "Filter input data.")
        ("exportScanPositions", value<bool>()->default_value(false), "Exports the original scan positions to 'scanpositions.txt'.")
	    ("k", value<int>()->default_value(1), "k neighborhood for filtering.")
	    ("sigma", value<float>()->default_value(1.0), "Deviation for outlier filter.")
	    ("targetSize", value<int>()->default_value(0), "Target size (reduction) for the input scans.")
        ("transformBefore", value<bool>()->default_value(false), "Transform the scans before frames/pose-transformation.")
	    ("rPos,r", value<int>()->default_value(-1), "Position of the red color component in the input data lines. (-1) means no color information")
	    ("gPos,g", value<int>()->default_value(-1), "Position of the green color component in the input data lines. (-1) means no color information")
	    ("bPos,b", value<int>()->default_value(-1), "Position of the blue color component in the input data lines. (-1) means no color information")
        ("start,s", value<int>()->default_value(0), "start at scan NR")
        ("end,e", value<int>()->default_value(0), "end at scan NR")
		("voxelSize,v", value<double>()->default_value(0.1), "Voxel size for octree reduction")
		("minPointsPerVoxel", value<size_t>()->default_value(5), "Minimum number of points per voxel in octree reduction")
		("scanPrefix", value<std::string>()->default_value("scan"), "Prexfix for scan files. E.g., scan for using scan001, scan002 etc.")
		("posePrefix", value<std::string>()->default_value("scan"), "Prexfix for files with 4x4 pose estimation in row-majow format. E.g., pose for using pose001, pose002 etc.")
		("scanExtension", value<std::string>()->default_value(".3d"), "File extension for parsed files containing point cloud data")
		("poseExtension", value<std::string>()->default_value(".dat"), "File extension for parsed files containing pose estimates")
		("convertToLVR", value<bool>()->default_value(false), "Convert a file in SLAM coordinates to LVR coordinates")
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

bool Options::convertToLVR() const
{
	return m_variables["convertToLVR"].as<bool>();
}


bool   Options:: exportScanPositions() const
{
    return m_variables["exportScanPositions"].as<bool>();
}

bool	Options::filter() const
{
	return m_variables["filter"].as<bool>();
}

bool    Options::transformBefore() const
{
    return m_variables["transformBefore"].as<bool>();
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

double  Options::getVoxelSize() const
{
	return m_variables["voxelSize"].as<double>();
}

size_t  Options::getMinPointsPerVoxel() const
{
	return m_variables["minPointsPerVoxel"].as<size_t>();
}


Options::~Options() {
	// TODO Auto-generated destructor stub
}

} // namespace reconstruct
