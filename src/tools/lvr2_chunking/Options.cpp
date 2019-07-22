/**
 * Copyright (c) 2019, University Osnabrück
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

 /**
 * Options.cpp
 *
 * @date 22.07.2019
 * @author Marcel Wiegand
 */

#include "Options.hpp"

#include <iostream>

namespace chunking {

using std::cout;
using std::endl;
using boost::program_options::value;
using boost::program_options::command_line_parser;

Options::Options(int argc, char** argv)
    : m_descr("Supported options")
{
    // Create option descriptions
    m_descr.add_options()
        ("help", "Produce help message")
        ("inputFile", value<string>(), "Input file name.")
    ;

    // Parse command line and generate variables map
    store(
          command_line_parser(argc, argv).options(m_descr).positional(m_posDescr).run(),
          m_variables);
    notify(m_variables);
}

bool Options::printUsage() const
{
    if (m_variables.count("help"))
    {
        cout << m_descr << endl;
        return true;
    }

    if (!m_variables.count("inputFile"))
    {
        cout << "Error: You must specify an input file." << endl;
        cout << endl;
        cout << m_descr << endl;
        return true;
    }

    return false;
}

string Options::getInputFile() const
{
    return m_variables["inputFile"].as<string>();
}

Options::~Options() {
    // TODO Auto-generated destructor stub
}

} // namespace chunking
