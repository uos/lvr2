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
#include "lvr2/io/hdf5/HDF5FeatureBase.hpp"
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
    if (options.getLoad())
    {
        if (boost::filesystem::exists(options.getChunkedMesh()))
        {
            // loading a hdf5 file and extracting the chunks for a given bounding box
            lvr2::ChunkManager chunkLoader(options.getChunkedMesh(), options.getCacheSize());

            // TODO: remove tmp test later
            // beginn: tmp test of extractArea method for dat/scan.pts with chunkSize 200
            if (!boost::filesystem::exists("area"))
            {
                boost::filesystem::create_directories("area");
            }
            lvr2::BoundingBox<lvr2::BaseVector<float>> area(
                lvr2::BaseVector<float>(options.getXMin(), options.getYMin(), options.getZMin()),
                lvr2::BaseVector<float>(options.getXMax(), options.getYMax(), options.getZMax()));
            // end: tmp test of extractArea method

//            lvr2::ModelFactory::saveModel(
//                lvr2::ModelPtr(new lvr2::Model(chunkLoader.extractArea(area))), "area.ply");
        }
    }
    else
    {
        // saving a mesh as multiple chunked meshes in an hdf5 file
        boost::filesystem::path outputPath = boost::filesystem::absolute(options.getOutputDir());
        if (!boost::filesystem::is_directory(outputPath))
        {
            boost::filesystem::create_directories(outputPath);
        }

        float size = options.getChunkSize();
        float maxChunkOverlap = options.getMaxChunkOverlap();


        // Check extension
        std::vector<std::string> files = options.getInputFile();
        boost::filesystem::path selectedFile(files[0]);
        std::string extension = selectedFile.extension().string();
        lvr2::MeshBufferPtr meshBuffer;
        if (extension == ".h5")
        {
            using HDF5MeshToolIO = lvr2::Hdf5IO<lvr2::hdf5features::ArrayIO,
                                                lvr2::hdf5features::ChannelIO,
                                                lvr2::hdf5features::VariantChannelIO,
                                                lvr2::hdf5features::MeshIO>;
            HDF5MeshToolIO hdf5;
            hdf5.open(files[0]);
            meshBuffer = hdf5.loadMesh(options.getMeshGroup());
        }
        else // use model reader
        {
            std::vector<lvr2::MeshBufferPtr> meshes;
            std::vector<std::string> layers;
            for(size_t i = 0; i < files.size(); ++i)
            {
                lvr2::ModelPtr model = lvr2::ModelFactory::readModel(files[i]);
                layers.push_back(std::string("mesh") + std::to_string(i));
                meshBuffer = model->m_mesh;
                if (meshBuffer)
                {
                    meshes.push_back(meshBuffer);
                }

            }
         lvr2::ChunkManager chunker(meshes, size, maxChunkOverlap, outputPath.string(), layers);
        }
    }
    return EXIT_SUCCESS;
}
