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
 *  Created on: Nov 21, 2010
 *      Author: Thomas Wiemann
 */

#include "Options.hpp"
#include "lvr2/config/lvropenmp.hpp"

#include <iostream>
#include <fstream>

namespace std
{
  std::ostream& operator<<(std::ostream &os, const std::vector<std::string> &vec) 
  {    
    for (auto item : vec) 
    { 
      os << item << " "; 
    } 
    return os; 
  }
} 

namespace ground_level_extractor{

using namespace boost::program_options;

Options::Options(int argc, char** argv)
    : BaseOption(argc, argv)
{
    // Create option descriptions
    m_descr.add_options()
        ("help", "Produce help message")
        ("inputFile", value< string >(), "Input file name. Supported formats are ASCII (.pts, .xyz) and .ply")
        ("outputFile", value< std::vector<string> >()->multitoken()->default_value(std::vector<string>{"dtm_mesh.ply", "dtm_mesh.obj"}), "Output file name. Supported formats are ASCII .obj and binary .ply")
        ("outputFolder", value< string >()->default_value(""), "Set a Folder to save your models in.")
        ("extractionMethod",value<string>(&m_method)->default_value("NN"),"Choose between three different methods to extract a DTM: NN(Nearest Neighbor), IMA (Improved Moving Average), THM (Threshold Method).")
        ("resolution", value<float>(&m_resolution)->default_value(1), "Set a resolution. Default is set to 1 and decreasing the value increases the number of points.")
        ("inputGeoTIFF", value< string >()->default_value(""), "Provide a GeoTIFF. Its data will be extracted and projected onto the model. You need to chose the bands you want to display.")
        ("inputReferencePairs", value< string >()->default_value(""),"Input file with reference coordinate pairs.")
        ("startingBand", value<int>(&m_startingBand)->default_value(1),"Set the first you want to extract.")
        ("numberOfBands", value<int>(&m_numberBands)->default_value(1),"Set the number of Bands you want to extract from the TIFF, includuing the starting band. You can chose between 1 or 3 Bands.")
        ("targetSystem", value<string>(&m_target)->default_value(""),"Name the coordinate system you want the model (and GeoTIFF) to be transformed into. This only works if you have provided a sufficient amount of reference points.")
        ("colorScale", value<string>(&m_colorScale)->default_value("JET"), "Set the color scale you want the Band or Heigh Difference Texture to be shown in.")
        ("numberNeighbors", value<int>(&m_numberNeighbors)->default_value(1), "Set the Number of Neighbors used by NN and IMA.")
        ("minRadius",value<float>(&m_minRadius)->default_value(0), "Set the minimum Radius used in IMA.")
        ("maxRadius",value<float>(&m_maxRadius)->default_value(1), "Set the maximum Radius used in IMA.")
        ("radiusSteps",value<int>(&m_radiusSteps)->default_value(100), "Set the amount of steps to reach the maximum Radius.")
        ("swSize", value<int>(&m_swSize)->default_value(3), "Size of the small window (x*x) in THM.")
        ("swThreshold", value<float>(&m_swThreshold)->default_value(1),"Threshold for the small window in THM.")
        ("lwSize", value<int>(&m_lwSize)->default_value(3), "Size of the large window (x*x) in THM.")
        ("lwThreshold", value<float>(&m_lwThreshold)->default_value(3),"Threshold for the large window in THM.")
        ("slopeThreshold", value<float>(&m_slopeThreshold)->default_value(30),"Threshold for the slope's angle in THM.");

    setup();
}

string Options::getInputFileName() const
{
    return (m_variables["inputFile"].as< string >());
}

string Options::getOutputFileName() const
{
    return getOutputFileNames()[0];
}

std::vector<string> Options::getOutputFileNames() const
{
    return  m_variables["outputFile"].as< std::vector<string> >();
}

string Options::getOutputFolderName() const
{
    return m_variables["outputFolder"].as<string>();
}

string Options::getExtractionMethod() const
{
    return m_variables["extractionMethod"].as<string>();
}

float Options::getResolution() const
{
    return m_variables["resolution"].as<float>();
}

string Options::getInputGeoTIFF() const
{
    return m_variables["inputGeoTIFF"].as<string>();
}

string Options::getInputReferencePairs() const
{
    return m_variables["inputReferencePairs"].as<string>();
}

int Options::getStartingBand() const
{
    return m_variables["startingBand"].as<int>();
}

int Options::getNumberOfBands() const
{
    return m_variables["numberOfBands"].as<int>();
}

string Options::getTargetSystem() const
{
    return m_variables["targetSystem"].as<string>();
}

string Options::getColorScale() const
{
    return m_variables["colorScale"].as<string>();
}

int Options::getNumberNeighbors() const
{
    return m_variables["numberNeighbors"].as<int>();
}

float Options::getMinRadius() const
{
    return m_variables["minRadius"].as<float>();
}

float Options::getMaxRadius() const
{
    return m_variables["maxRadius"].as<float>();
}

int Options::getRadiusSteps() const
{
    return m_variables["radiusSteps"].as<int>();
}

int Options::getSWSize() const
{
    return m_variables["swSize"].as<int>();
}

float Options::getSWThreshold() const
{
    return m_variables["swThreshold"].as<float>();
}

int Options::getLWSize() const
{
    return m_variables["lwSize"].as<int>();
}

float Options::getLWThreshold() const
{
    return m_variables["lwThreshold"].as<float>();
}

float Options::getSlopeThreshold() const
{
    return m_variables["slopeThreshold"].as<float>();
}

bool Options::printUsage() const {
        if (m_variables.count("help"))
        {
            cout << endl;
            cout << m_descr << endl;
            return true;
        }
        else if (!m_variables.count("inputFile"))
        {
            cout << "Error: You must specify an input file." << endl;
            cout << endl;
            cout << m_descr << endl;
            return true;
        }
        return false;
    }


Options::~Options() {
    // TODO Auto-generated destructor stub
}

} // namespace ground_level_extractor
