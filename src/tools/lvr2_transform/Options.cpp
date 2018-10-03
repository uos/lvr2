/* Copyright (C) 2012 Uni Osnabrück
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
 * @date 2012-08-23
 * @author Christian Wansart <cwansart@uos.de>
 * @author Thomas Wiemann
 */

#include "Options.hpp"
//#include <omp.h>

#define vc(x) ( m_variables.count(x) )
#define srtOR ( !vc("sx") && !vc("sy") && !vc("sz") && \
                !vc("rx") && !vc("ry") && !vc("rz") && \
                !vc("tx") && !vc("ty") && !vc("tz") )

namespace transform {

Options::Options(int argc, char** argv) : m_descr("Supported options")
{
  // Create option descriptions

  m_descr.add_options()
    ("help", "Produce help message")
    ("inputFile,i",     value< string >(),          "Input file name.")
    ("outputFile,o",    value< string >(),          "Output file name. Don't forget to append a file extension.")
    ("transformFile,t", value< string >(),          "*.pose or *.frame file for transformation")

    ("sx",              value< float >(),           "Scale in x axis.")
    ("sy",              value< float >(),           "Scale in y axis.")
    ("sz",              value< float >(),           "Scale in z axis.")
//    ("scale,s",         value< vector< float >>() "Scale in all axis. (Format: x,y,z; e.g. --scale 1.0,4.2,0.4)")

    ("rx",              value< float >(),           "Rotation in x axis.")
    ("ry",              value< float >(),           "Rotation in y axis.")
    ("rz",              value< float >(),           "Rotation in z axis.")
//    ("rotation",        value< vector< float >>() "Rotation in all axis. (Format: x,y,z; e.g. --rotation 4.0,5.0,0.0)")

    ("tx",              value< float >(),           "Transform in x axis.")
    ("ty",              value< float >(),           "Transform in y axis.")
    ("tz",              value< float >(),           "Transform in z axis.")
//    ("translate",       value< vector< float >>(),  "Transform in all axis. (Format: x,y,z; e.g.: --translate 6.0,0.0,1.2)")
    ;

  // Parse command line and generate variables map
  store(command_line_parser(argc, argv).options(m_descr).positional(m_pdescr).run(), m_variables);
  notify(m_variables);

  if(m_variables.count("help")) {
    cout<< m_descr << endl;
  }

}

string Options::getInputFile() const
{
  return (m_variables["inputFile"].as< string >());
}

string Options::getOutputFile() const
{
  return (m_variables["outputFile"].as< string >());
}

string Options::getTransformFile() const
{
  return (m_variables["transformFile"].as< string >());
}

bool Options::anyTransformFile() const
{
  return (m_variables.count("transformFile"));
}

float Options::getScaleX() const
{
  return (m_variables["sx"].as< float >());
}

float Options::getScaleY() const
{
  return (m_variables["sy"].as< float >());
}

float Options::getScaleZ() const
{
  return (m_variables["sz"].as< float >());
}

float Options::getRotationX() const
{
  return (m_variables["rx"].as< float >());
}

float Options::getRotationY() const
{
  return (m_variables["ry"].as< float >());
}

float Options::getRotationZ() const
{
  return (m_variables["rz"].as< float >());
}

float Options::getTranslationX() const
{
  return (m_variables["tx"].as< float >());
}

float Options::getTranslationY() const
{
  return (m_variables["ty"].as< float >());
}

float Options::getTranslationZ() const
{
  return (m_variables["tz"].as< float >());
}

bool Options::anyScale() const
{
  return (anyScaleX() || anyScaleY() || anyScaleZ());
}

bool Options::anyScaleX() const
{
  return (m_variables.count("sx"));
}

bool Options::anyScaleY() const
{
  return (m_variables.count("sy"));
}

bool Options::anyScaleZ() const
{
  return (m_variables.count("sz"));
}

bool Options::anyRotation() const
{
  return (anyRotationX() || anyRotationY() || anyRotationZ());
}

bool Options::anyRotationX() const
{
  return (m_variables.count("rx"));
}

bool Options::anyRotationY() const
{
  return (m_variables.count("ry"));
}

bool Options::anyRotationZ() const
{
  return (m_variables.count("rz"));
}

bool Options::anyTranslation() const
{
  return (anyTranslationX() || anyTranslationY() || anyTranslationZ());
}

bool Options::anyTranslationX() const
{
  return (m_variables.count("tx"));
}

bool Options::anyTranslationY() const
{
  return (m_variables.count("ty"));
}

bool Options::anyTranslationZ() const
{
  return (m_variables.count("tz"));
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
  
  if(!m_variables.count("outputFile"))
  {
    cout << "Error: You must specify an output file." << endl;
    cout << endl;
    cout << m_descr << endl;
    return true;
  }
 
  // TODO:  Need to check for s, t or r
  //        add notice that s, t and r will be ignored if transformFile is given
  if(!m_variables.count("transformFile") && srtOR)
  {
    cout << "Error: You must specify a transform file or sx, sy, sz, rx, ry, rz, tx, ty, tz." << endl;
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
}

} // namespace transform

