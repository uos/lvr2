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
 * Options.h
 *
 *  Created on: Aug 23 2012
 *      Author: Thomas Wiemann
 */

#ifndef OPTIONS_H_
#define OPTIONS_H_

#include "lvr2/config/BaseOption.hpp"
#include "lvr2/util/ScanSchemaUtils.hpp"
#include "lvr2/util/ScanProjectUtils.hpp"

#include <boost/program_options.hpp>
#include <iostream>
#include <string>
#include <vector>

using std::cout;
using std::endl;
using std::ostream;
using std::string;
using std::vector;

namespace scanproject_parser
{

using namespace boost::program_options;

/**
 * @brief A class to parse the program options for the reconstruction
 * 		  executable.
 */
class Options : public lvr2::BaseOption
{
  public:
    /**
     * @brief 	Ctor. Parses the command parameters given to the main
     * 		  	function of the program
     */
    Options(int argc, char** argv);
    virtual ~Options();

    string getInputSource() const { return m_variables["inputSource"].as<string>(); }
    string getOutputSource() const { return m_variables["outputSource"].as<string>(); }
    string getOutputSchema() const { return m_variables["outputSchema"].as<string>(); }
    string getInputSchema() const { return m_variables["inputSchema"].as<string>(); }

    bool printStructure() const { return m_variables.count("printStructure"); }
    bool convert() const {return m_variables.count("convert");}
    bool computeNormals() const {return m_variables.count("computeNormals");}

    size_t kn() const {return m_variables["kn"].as<size_t>();}
    size_t ki() const {return m_variables["ki"].as<size_t>();}

    std::vector<size_t> scanPositions()
    {
      if (m_variables.count("scanpositions"))
      {
        return m_variables["scanpositions"].as<std::vector<size_t>>();
      }
      else
      {
        return std::vector<size_t>();
      }
    }

private :
    /// The internally used variable map
    variables_map m_variables;

    /// The internally used option description
    options_description m_descr;

    /// The internally used positional option desription
    positional_options_description m_pdescr;

    /// The help string that displays all implemented scan schemas
    std::string getSupportedSchemasHelpString();
};



} // namespace hdf5tool2

#endif /* OPTIONS_H_ */
