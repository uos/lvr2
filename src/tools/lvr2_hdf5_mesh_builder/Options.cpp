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

#include "Options.hpp"

namespace hdf5meshtool
{

Options::Options(int argc, char** argv) : m_descr("Supported options")
{

	// Create option descriptions

	m_descr.add_options()
            ("help", "Produce help message")
            ("inputFile,i", value<string>()->default_value("triangle_mesh.ply"), "Input mesh file.")
            ("outputFile,o", value<string>()->default_value("mesh.h5"), "Output file.")
            ("meshName,m", value<string>()->default_value("mesh"), "The name of the mesh to write")
            ("edgeCollapse,e", value<size_t>()->default_value(0), "Edge collapse reduction algorithm, the number of edges to collapse.")
            ("localRadius,r", value<float>()->default_value(0.3), "The local radius used for roughness and height differences computation.")
            ;


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
