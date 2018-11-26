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

#include "Options.hpp"

namespace image_normals
{

Options::Options(int argc, char** argv) : m_descr("Supported options")
{

	// Create option descriptions

	m_descr.add_options()
	("help", "Produce help message")
    ("inputFile", value< vector<string> >(), "Input file name. ")
    ("minH",            value<int>(&m_minH)->default_value(0),          "Minimal horizontal opening angle.")
    ("maxH",            value<int>(&m_maxH)->default_value(360),        "Maximal horizontal opening angle.")
    ("minV",            value<int>(&m_minV)->default_value(-90),        "Minimal vertical opening angle.")
    ("maxV",            value<int>(&m_maxV)->default_value(90),         "Maximal vertical opening angle.")
    ("minZ",            value<float>(&m_minZ)->default_value(0),        "Minimal depth value.")
    ("maxZ",            value<float>(&m_maxZ)->default_value(1e6),      "Maximal depth value.")
    ("maxZimg",         value<float>(&m_maxZimg)->default_value(1e6),   "Maximal depth value in depth image.")
    ("minZimg",         value<float>(&m_maxZimg)->default_value(0),     "Maximal depth value in depth image.")
    ("img",             value<string>(&m_imageOut)->default_value("panorama.pgm"), "Output file for projection image.")
    ("imageWidth,w",    value<int>(&m_width)->default_value(2800),      "Image width.")
    ("imageHeight,h",   value<int>(&m_height)->default_value(1000),     "Image height.")
    ("regionWidth,i",    value<int>(&m_width)->default_value(5),      "Width of the nearest neighbor region of a pixel for normal estimation.")
    ("regionHeight,j",   value<int>(&m_height)->default_value(5),     "Height of the nearest neighbor region of a pixel for normal estimation.")
    ("optimize,o",      "Optimize image aspect ratio.")
    ("system,s",        value<string>(&m_system)->default_value("NATIVE"), "The coordinate system in which the points are stored. Use NATIVE to interpret the points as they are. Use SLAM6D for scans in 3dtk's coordinate system and UOS for scans that where taken with a tilting laser scanner at Osnabrueck University.")
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

