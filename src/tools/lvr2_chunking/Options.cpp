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

namespace chunking
{

using boost::program_options::command_line_parser;
using boost::program_options::value;
using std::cout;
using std::endl;

Options::Options(int argc, char** argv) : m_descr("Supported options")
{
    // Create option descriptions
    m_descr.add_options()("help",
                          "Produce help message")("inputFile", value<std::vector<string>>()->multitoken(), "Input file names.")(
        "outputDir", value<string>(), "Output directory name.")(
        "chunkSize", value<float>()->default_value(10.0f), "Side length of chunks.")(
        "maxChunkOverlap", value<float>()->default_value(0.1f),
        "maximum allowed overlap between chunks relative to the chunk size.")(
        "chunkedMesh", value<string>(), "Chunked mesh hdf5-file name.")(
        "load", value<bool>()->default_value(false), "Set this value to true, if you want to load an hdf5-file")(
        "x_min", value<float>()->default_value(0.0f), "bounding box minimum value in x-dimension")(
        "y_min", value<float>()->default_value(0.0f), "bounding box minimum value in y-dimension")(
        "z_min", value<float>()->default_value(0.0f), "bounding box minimum value in z-dimension")(
        "x_max", value<float>()->default_value(10.0f), "bounding box maximum value in x-dimension")(
        "y_max", value<float>()->default_value(10.0f), "bounding box maximum value in y-dimension")(
        "z_max", value<float>()->default_value(10.0f), "bounding box maximum value in z-dimension")(
        "cacheSize", value<int>()->default_value(200), "while loading the maximum number of chunks in RAM")(
        "meshName", value<std::string>()->default_value(""), "group name of the mesh if the HDF5 contains multiple meshes");

    // Parse command line and generate variables map
    store(command_line_parser(argc, argv).options(m_descr).positional(m_posDescr).run(),
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

    if (!m_variables.count("inputFile") && !getLoad())
    {
        cout << "Error: You must specify an input file." << endl;
        cout << endl;
        cout << m_descr << endl;
        return true;
    }

    if(getLoad() && !m_variables.count("chunkedMesh"))
    {
        cout << "Error: You must specify an hdf5-file (\"chunkedMesh\") when loading." << endl;
        cout << endl;
        cout << m_descr << endl;
        return true;
    }

    return false;
}

std::vector<string> Options::getInputFile() const
{

    return m_variables["inputFile"].as<std::vector<string>>();
}

string Options::getOutputDir() const
{
    return m_variables["outputDir"].as<string>();
}

float Options::getChunkSize() const
{
    return m_variables["chunkSize"].as<float>();
}

float Options::getMaxChunkOverlap() const
{
    return m_variables["maxChunkOverlap"].as<float>();
}
string Options::getChunkedMesh() const
{
    return m_variables["chunkedMesh"].as<string>();
}
bool Options::getLoad() const
{
    return m_variables["load"].as<bool>();
}

float Options::getXMin() const
{
    return m_variables["x_min"].as<float>();
}
float Options::getYMin() const
{
    return m_variables["y_min"].as<float>();
}
float Options::getZMin() const
{
    return m_variables["z_min"].as<float>();
}
float Options::getXMax() const
{
    return m_variables["x_max"].as<float>();
}
float Options::getYMax() const
{
    return m_variables["y_max"].as<float>();
}
float Options::getZMax() const
{
    return m_variables["z_max"].as<float>();
}
int Options::getCacheSize() const
{
    int size = m_variables["cacheSize"].as<int>();
    if(size > 0)
    {
        return size;
    }
    return 200;
}
std::string Options::getMeshGroup() const
{
    return m_variables["meshName"].as<std::string>();
}

Options::~Options()
{
    // TODO Auto-generated destructor stub
}

} // namespace chunking
