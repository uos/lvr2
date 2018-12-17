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

namespace hdf5tool
{

Options::Options(int argc, char** argv) : m_descr("Supported options")
{

	// Create option descriptions

	m_descr.add_options()
            ("help", "Produce help message")
            ("dataDir", value<string>()->default_value("./"), "Directory with hyperspectral data.")
            ("nch, n", value<int>()->default_value(150), "Number of spectral PNGs in image folder.")
            ("hsp_chunk_0", value<size_t>()->default_value(50), "Dim 0 of HSP image chunks.")
            ("hsp_chunk_1", value<size_t>()->default_value(50), "Dim 1 of HSP image chunks.")
            ("hsp_chunk_2", value<size_t>()->default_value(50), "Dim 2 of HSP image chunks.")
            ("addAnnotations", value<int>()->default_value(1), "Add spectral annotation channels");


	// Parse command line and generate variables map
	store(command_line_parser(argc, argv).options(m_descr).positional(m_pdescr).run(), m_variables);
	notify(m_variables);

  if(m_variables.count("help")) {
    ::std::cout<< m_descr << ::std::endl;
    exit(-1);
  }


}

Options::~Options() {
	// TODO Auto-generated destructor stub
}

} // namespace reconstruct
