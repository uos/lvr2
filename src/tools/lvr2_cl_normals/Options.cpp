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

namespace cl_normals
{

Options::Options(int argc, char** argv) : m_descr("Supported options")
{

    // Create option descriptions

    m_descr.add_options()
    ("help", "Produce help message")
    ("inputFile", value< vector<string> >(), "Input file name. ")
    ("outputFile,o",    value<string>(&m_outputFile)->default_value("normals.ply"), "Output file name.")
    ("ransac", "Set this flag for RANSAC based normal estimation.")
    ("pca", "Set this flag for RANSAC based normal estimation.")
    ("kn", value<int>(&m_kn)->default_value(10), "Number of normals used for normal estimation")
    ("ki", value<int>(&m_ki)->default_value(10), "Number of normals used for normal interpolation")
    ("kd", value<int>(&m_kd)->default_value(5), "Number of normals used for distance calculation")
    ("flipx", value<float>(&m_flipx)->default_value(100000.0), "Flippoint x" )
    ("flipy", value<float>(&m_flipy)->default_value(100000.0), "Flippoint y" )
    ("flipz", value<float>(&m_flipz)->default_value(100000.0), "Flippoint z" )
    ("reconstruct,r","Reconstruct after normal calculation")
    ("exportPointNormals,e","save Pointnormals before reconstruction")
    ("voxelsize,v",value<float>(&m_voxelsize)->default_value(10.0),"voxelsize for marching cubes")
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
