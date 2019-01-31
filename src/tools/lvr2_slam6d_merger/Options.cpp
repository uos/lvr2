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

namespace slam6dmerger
{

Options::Options(int argc, char** argv) : m_descr("Supported options")
{

	// Create option descriptions

	m_descr.add_options()
		("help", "Produce help message")
		("inputDir", value<string>()->default_value("./"), "A directory containing several scan files for batch conversion.")
        ("mergeDir", value<string>()->default_value("./"), "A directory containing scans to merge into the project defined in inputDir")
        ("outputDir", value<string>()->default_value("./"), "The target directory for the merge scan data.")
        ("transformFile", value<string>()->default_value("./"), "A text file containing the transformation between the coordinates systems of inputDir and mergeDir")
        ("start,s", value<int>()->default_value(0), "start at scan NR in mergeDir")
        ("end,e", value<int>()->default_value(0), "end at scan NR in mergeDir")
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


string 	Options::getInputDir() const
{
	return m_variables["inputDir"].as<string>();
}

string 	Options::getMergeDir() const
{
    return m_variables["mergeDir"].as<string>();
}

string 	Options::getOutputDir() const
{
	return m_variables["outputDir"].as<string>();
}

string 	Options::getTransformFile() const
{
    return m_variables["transformFile"].as<string>();
}



Options::~Options() {
	// TODO Auto-generated destructor stub
}

} // namespace reconstruct
