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
 * Main.cpp
 *
 * @date 22.07.2019
 * @author Marcel Wiegand
 */

#include "Options.hpp"
#include "lvr2/algorithm/ChunkManager.hpp"
#include "lvr2/io/ModelFactory.hpp"

#include <boost/filesystem.hpp>
#include <iostream>
#include <string>

int main(int argc, char** argv)
{
    // =======================================================================
    // Parse and print command line parameters
    // =======================================================================
    // Parse command line arguments
    chunking::Options options(argc, argv);

    // Exit if options had to generate a usage message
    // (this means required parameters are missing)
    if (options.printUsage())
    {
        return EXIT_SUCCESS;
    }

    lvr2::ModelPtr model = lvr2::ModelFactory::readModel(options.getInputFile());

    boost::filesystem::path outputPath = boost::filesystem::absolute(options.getOutputDir());
    if (!boost::filesystem::is_directory(outputPath))
    {
        boost::filesystem::create_directories(outputPath);
    }

    float size = options.getChunkSize();
    float maxChunkOverlap = options.getMaxChunkOverlap();

    lvr2::ChunkManager chunker(model->m_mesh, size, maxChunkOverlap, outputPath.string());

    // TODO: remove tmp test later
    // beginn: tmp test of extractArea method for dat/scan.pts with chunkSize 200
    if (!boost::filesystem::exists("area"))
    {
        boost::filesystem::create_directories("area");
    }
    lvr2::BoundingBox<lvr2::BaseVector<float>> area(lvr2::BaseVector<float>(-50, 130, 105),
                                                    lvr2::BaseVector<float>(155, 194, 211));
    // end: tmp test of extractArea method

    lvr2::ModelFactory::saveModel(lvr2::ModelPtr(new lvr2::Model(chunker.extractArea(area))),
                                  "area.ply");

    return EXIT_SUCCESS;
}
