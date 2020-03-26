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
#include "lvr2/config/lvropenmp.hpp"
#include <random>
#include <string>
#include <lvr2/io/hdf5/ScanIO.hpp>
#include <boost/filesystem.hpp>
#include "lvr2/io/hdf5/HDF5FeatureBase.hpp"
#include "lvr2/io/hdf5/ScanProjectIO.hpp"
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
// using ScanHDF5IO = lvr2::Hdf5Build<lvr2::hdf5features::ScanIO>;

using BaseHDF5IO = lvr2::Hdf5IO<>;

// Extend IO with features (dependencies are automatically fetched)
using HDF5IO = BaseHDF5IO::AddFeatures<lvr2::hdf5features::ScanProjectIO>;

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

    OpenMPConfig::setNumThreads(options.getNumThreads());

    LargeScaleReconstruction<Vec> lsr(options.getVoxelSizes(), options.getBGVoxelsize(), options.getScaling(),
                                      options.getNodeSize(), options.getPartMethod(), options.getKi(), options.getKd(), options.getKn(),
                                      options.useRansac(), options.getFlippoint(), options.extrude(), options.getDanglingArtifacts(),
                                      options.getCleanContourIterations(), options.getFillHoles(), options.optimizePlanes(),
                                      options.getNormalThreshold(), options.getPlaneIterations(), options.getMinPlaneSize(), options.getSmallRegionThreshold(),
                                      options.retesselate(), options.getLineFusionThreshold(), options.getBigMesh(), options.getDebugChunks(), options.useGPU());

    


    ScanProjectEditMarkPtr project(new ScanProjectEditMark);
    std::shared_ptr<ChunkHashGrid> cm;
    BoundingBox<Vec> boundingBox;

    HDF5IO hdf;

    //reconstruction from hdf5
    if (extension == ".h5")
    {
        // loadAllPreviewsFromHDF5(in, *project->project.get());
        HDF5IO hdf;
        hdf.open(in);
        ScanProjectPtr scanProjectPtr = hdf.loadScanProject();
        project->project = scanProjectPtr;

        for (int i = 0; i < project->project->positions.size(); i++)
        {
            project->changed.push_back(true);
        }
        cm = std::shared_ptr<ChunkHashGrid>(new ChunkHashGrid(in, 50, boundingBox, options.getChunkSize()));
    }
    else
    {

        ScanProject dirScanProject;
        bool importStatus = loadScanProject(in, dirScanProject);
        //reconstruction from ScanProject Folder
        if(importStatus) {
            project->project = make_shared<ScanProject>(dirScanProject);
            std::vector<bool> init(dirScanProject.positions.size(), true);
            project->changed = init;
        }
        //reconstruction from a .ply file
        else if(!boost::filesystem::is_directory(selectedFile))
        {
            project->project = ScanProjectPtr(new ScanProject);
            ModelPtr model = ModelFactory::readModel(in);
            ScanPtr scan(new Scan);

            scan->points = model->m_pointCloud;
            ScanPositionPtr scanPosPtr = ScanPositionPtr(new ScanPosition());
            scanPosPtr->scans.push_back(scan);
            project->project->positions.push_back(scanPosPtr);
            project->changed.push_back(true);
        }
        //reconstruction from a folder of .ply files
        else{
            project->project = ScanProjectPtr(new ScanProject);
            boost::filesystem::directory_iterator it{in};
            while (it != boost::filesystem::directory_iterator{})
            {
                cout << it->path().string() << endl;
                string ext = it->path().extension().string();
                if(ext == ".ply")
                {
                    ModelPtr model = ModelFactory::readModel(it->path().string());
                    ScanPtr scan(new Scan);

                    scan->points = model->m_pointCloud;
                    ScanPositionPtr scanPosPtr = ScanPositionPtr(new ScanPosition());
                    scanPosPtr->scans.push_back(scan);
                    project->project->positions.push_back(scanPosPtr);
                    project->changed.push_back(true);
                }
                it++;
            }


        }

        cm = std::shared_ptr<ChunkHashGrid>(new ChunkHashGrid("chunked_mesh.h5", 50, boundingBox, options.getChunkSize()));
    }

    BoundingBox<Vec> bb;
    // reconstruction with diffrent methods
    if(options.getPartMethod() == 1)
    {
        int x = lsr.mpiChunkAndReconstruct(project, bb, cm);
    }
    else
    {
        int x = lsr.mpiAndReconstruct(project);
    }

    // reconstruction of .ply for diffrent voxelSizes
    if(options.getDebugChunks())
    {

        for (int i; i < options.getVoxelSizes().size(); i++) {
            lsr.getPartialReconstruct(bb, cm, options.getVoxelSizes()[i]);
        }
    }

    cout << "Program end." << endl;

    return 0;
}
