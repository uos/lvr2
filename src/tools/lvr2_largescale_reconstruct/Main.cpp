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

#include "LargeScaleOptions.hpp"
#include "lvr2/reconstruction/LargeScaleReconstruction.hpp"
#include "lvr2/algorithm/GeometryAlgorithms.hpp"
#include <algorithm>
#include <iostream>
#include "lvr2/geometry/BaseVector.hpp"
#include <random>
#include <string>
#include <lvr2/io/hdf5/ScanIO.hpp>
#include "lvr2/io/GHDF5IO.hpp"
#include "lvr2/io/ScanIOUtils.hpp"

using std::cout;
using std::endl;
using namespace lvr2;

#if defined CUDA_FOUND
#define GPU_FOUND

#include "lvr2/reconstruction/cuda/CudaSurface.hpp"

typedef CudaSurface GpuSurface;
#elif defined OPENCL_FOUND
#define GPU_FOUND

#include "lvr2/reconstruction/opencl/ClSurface.hpp"


typedef ClSurface GpuSurface;
#endif

using Vec = lvr2::BaseVector<float>;
using ScanHDF5IO = lvr2::Hdf5Build<lvr2::hdf5features::ScanIO>;
int main(int argc, char** argv)
{
    // =======================================================================
    // Parse and print command line parameters
    // =======================================================================
    // Parse command line arguments
    LargeScaleOptions::Options options(argc, argv);

    options.printLogo();

    // Exit if options had to generate a usage message
    // (this means required parameters are missing)
    if (options.printUsage())
    {
        return EXIT_SUCCESS;
    }

    std::cout << options << std::endl;

    string in = options.getInputFileName()[0];

    boost::filesystem::path selectedFile(in);
    string extension = selectedFile.extension().string();


    LargeScaleReconstruction<Vec> lsr(options.getInputFileName()[0], options.getVoxelsize(), options.getBGVoxelsize(), options.getScaling(), options.getGridSize(),
                                      options.getNodeSize(), options.getVGrid(), options.getKi(), options.getKd(), options.getKn(), options.useRansac(), options.extrude(),
                                      options.getDanglingArtifacts(), options.getCleanContourIterations(), options.getFillHoles(), options.optimizePlanes(),
                                      options.getNormalThreshold(), options.getPlaneIterations(), options.getMinPlaneSize(), options.getSmallRegionThreshold(),
                                      options.retesselate(), options.getLineFusionThreshold(), options.getBigMesh(), options.getDebugChunks(), options.useGPU());

    ScanProjectEditMarkPtr project(new ScanProjectEditMark);
    project->project = ScanProjectPtr(new ScanProject);
    std::shared_ptr<ChunkHashGrid> cm;
    BoundingBox<Vec> boundingBox;

    if (extension == ".h5")
    {
        loadAllPreviewsFromHDF5(in, *project->project.get());

        for (int i = 0; i < project->project->positions.size(); i++)
        {
            project->changed.push_back(true);
        }
        cm = std::shared_ptr<ChunkHashGrid>(new ChunkHashGrid(in, 50, boundingBox, options.getGridSize()));
    }
    else
    {
        ModelPtr model = ModelFactory::readModel(in);
        Scan scan;

        scan.m_points = model->m_pointCloud;
        ScanPositionPtr scanPosPtr = ScanPositionPtr(new ScanPosition());
        scanPosPtr->scan = scan;
        project->project->positions = std::vector<ScanPositionPtr>();
        project->project->positions.push_back(scanPosPtr);
        project->changed.push_back(true);

        cm = std::shared_ptr<ChunkHashGrid>(new ChunkHashGrid("chunked_mesh.h5", 50, boundingBox, options.getGridSize()));
    }


    BoundingBox<Vec> bb;
    int x = lsr.mpiChunkAndReconstruct(project, bb, cm, "tsdf_values");


    //lsr.getPartialReconstruct(bb, cm, "tsdf_values");

    cout << "Program end." << endl;

    return 0;
}
