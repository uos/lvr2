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


#include <iostream>
#include <memory>
#include <tuple>
#include <stdlib.h>
#include <fstream>

#include <boost/optional.hpp>
#include <boost/shared_array.hpp>
#include <boost/smart_ptr/make_shared_array.hpp>
// TODO Delete
#include <chrono>

#include "lvr2/config/lvropenmp.hpp"

#include "lvr2/geometry/PMPMesh.hpp"
#include "lvr2/geometry/HalfEdgeMesh.hpp"
#include "lvr2/geometry/BaseVector.hpp"
#include "lvr2/geometry/Normal.hpp"
#include "lvr2/attrmaps/StableVector.hpp"
#include "lvr2/attrmaps/VectorMap.hpp"
#include "lvr2/algorithm/FinalizeAlgorithms.hpp"
#include "lvr2/algorithm/NormalAlgorithms.hpp"
#include "lvr2/algorithm/ColorAlgorithms.hpp"
#include "lvr2/geometry/BoundingBox.hpp"
#include "lvr2/algorithm/Tesselator.hpp"
#include "lvr2/algorithm/ClusterPainter.hpp"
#include "lvr2/algorithm/ClusterAlgorithms.hpp"
#include "lvr2/algorithm/CleanupAlgorithms.hpp"
#include "lvr2/algorithm/ReductionAlgorithms.hpp"
#include "lvr2/algorithm/Materializer.hpp"
#include "lvr2/algorithm/Texturizer.hpp"
#include "lvr2/reconstruction/AdaptiveKSearchSurface.hpp" // Has to be included before anything includes opencv stuff, see https://github.com/flann-lib/flann/issues/214 
#include "lvr2/reconstruction/CudaKSearchSurface.hpp"
#include "lvr2/algorithm/SpectralTexturizer.hpp"

#ifdef LVR2_USE_EMBREE
    #include "lvr2/algorithm/RaycastingTexturizer.hpp"
#endif

#include "lvr2/reconstruction/BilinearFastBox.hpp"
#include "lvr2/reconstruction/TetraederBox.hpp"
#include "lvr2/reconstruction/FastReconstruction.hpp"
#include "lvr2/reconstruction/PointsetSurface.hpp"
#include "lvr2/reconstruction/SearchTree.hpp"
#include "lvr2/reconstruction/SearchTreeFlann.hpp"
#include "lvr2/reconstruction/SearchTreeLBVH.hpp"
#include "lvr2/reconstruction/HashGrid.hpp"
#include "lvr2/reconstruction/PointsetGrid.hpp"
#include "lvr2/reconstruction/SharpBox.hpp"
#include "lvr2/types/PointBuffer.hpp"
#include "lvr2/types/MeshBuffer.hpp"
#include "lvr2/io/ModelFactory.hpp"
#include "lvr2/io/PlutoMapIO.hpp"
#include "lvr2/io/meshio/HDF5IO.hpp"
#include "lvr2/io/meshio/DirectoryIO.hpp"
#include "lvr2/util/Factories.hpp"
#include "lvr2/algorithm/GeometryAlgorithms.hpp"
#include "lvr2/algorithm/UtilAlgorithms.hpp"
#include "lvr2/algorithm/KDTree.hpp"
#include "lvr2/io/kernels/HDF5Kernel.hpp"
#include "lvr2/io/scanio/HDF5IO.hpp"
#include "lvr2/io/scanio/ScanProjectIO.hpp"
#include "lvr2/io/schema/ScanProjectSchema.hpp"
#include "lvr2/io/schema/ScanProjectSchemaHDF5.hpp"

#include "lvr2/geometry/BVH.hpp"

#include "lvr2/reconstruction/DMCReconstruction.hpp"

// TODO
#include "lvr2/util/Synthetic.hpp"

#include "Options.hpp"

#if defined LVR2_USE_CUDA
    #define GPU_FOUND

    #include "lvr2/reconstruction/cuda/CudaSurface.hpp"

    typedef lvr2::CudaSurface GpuSurface;
#elif defined LVR2_USE_OPENCL
    #define GPU_FOUND

    #include "lvr2/reconstruction/opencl/ClSurface.hpp"
    typedef lvr2::ClSurface GpuSurface;
#endif

using boost::optional;
using std::unique_ptr;
using std::make_unique;

using namespace lvr2;

using Vec = BaseVector<float>;
using PsSurface = lvr2::PointsetSurface<Vec>;


const char *path = "/home/tests/runtime_tests/";
std::ofstream myfile(path);


template <typename IteratorType>
IteratorType concatenate(
    IteratorType output_,
    IteratorType begin0,
    IteratorType end0,
    IteratorType begin1,
    IteratorType end1)
{
    output_ = std::copy(
        begin0,
        end0,
        output_);
    output_ = std::copy(
        begin1,
        end1,
        output_);

    return output_;
}


/**
 * @brief Merges two PointBuffers by copying the data into a new PointBuffer
 * 
 * The function does not modify its arguments, but its not possible to access the PointBuffers data
 * 
 * @param b0 A buffer to copy points from
 * @param b1 A buffer to copy points from
 * @return PointBuffer the merged result of b0 and b1
 */
PointBuffer mergePointBuffers(PointBuffer& b0, PointBuffer& b1)
{
    // number of points in new buffer
    PointBuffer::size_type npoints_total = b0.numPoints() + b1.numPoints();
    // new point array
    floatArr merged_points = floatArr(new float[npoints_total * 3]);

    auto output_it = merged_points.get();
    
    // Copy the points to the new array
    output_it = concatenate(
        output_it,
        b0.getPointArray().get(),
        b0.getPointArray().get() + (b0.numPoints() * 3),
        b1.getPointArray().get(),
        b1.getPointArray().get() + (b1.numPoints() * 3));

    // output iterator should be at the end of the array
    assert(output_it == merged_points.get() + (npoints_total * 3));

    PointBuffer ret(merged_points, npoints_total);

    // Copy colors 
    if (b0.hasColors() && b1.hasColors())
    {
        // nbytes of a color
        size_t w0, w1;
        b0.getColorArray(w0);
        b1.getColorArray(w1);
        if (w0 != w1)
        {
            panic("PointBuffer colors must have the same width!");
        }
        // Number of bytes needed for the colors. Assumes that both color widths are the same
        size_t nbytes = npoints_total * w0;
        ucharArr colors_total = ucharArr(new unsigned char[nbytes]);
        auto output_it = colors_total.get();

        output_it = concatenate(
            output_it,
            b0.getColorArray(w0).get(),
            b0.getColorArray(w0).get() + (b0.numPoints() * w0),
            b1.getColorArray(w1).get(),
            b1.getColorArray(w1).get() + (b1.numPoints() * w1)
        );
        
        ret.setColorArray(colors_total, npoints_total, w0);
    }

    // Copy normals
     if (b0.hasNormals() && b1.hasNormals())
    {
        // Number of bytes needed for the normals
        size_t nbytes = npoints_total * 3;
        floatArr normals_total = floatArr(new float[nbytes]);
        auto output_it = normals_total.get();

        output_it = concatenate(
            output_it,
            b0.getNormalArray().get(),
            b0.getNormalArray().get() + (b0.numPoints() * 3),
            b1.getNormalArray().get(),
            b1.getNormalArray().get() + (b1.numPoints() * 3)
        );
        
        ret.setNormalArray(normals_total,npoints_total);
    }

    return std::move(ret);
}

template <typename BaseVecT>
PointsetSurfacePtr<BaseVecT> loadPointCloud(const reconstruct::Options& options)
{   

    // Create a point loader object
    ModelPtr model = ModelFactory::readModel(options.getInputFileName());
    PointBufferPtr buffer;
    // Parse loaded data
    if (!model)
    {
        boost::filesystem::path selectedFile( options.getInputFileName());
        std::string extension = selectedFile.extension().string();
        std::string filePath = selectedFile.generic_path().string();

        if(selectedFile.extension().string() != ".h5") {
            cout << timestamp << "IO Error: Unable to parse " << options.getInputFileName() << endl;
            return nullptr;
        }
        cout << timestamp << "Loading h5 scanproject from " << filePath << endl;

        // create hdf5 kernel and schema 
        FileKernelPtr kernel = FileKernelPtr(new HDF5Kernel(filePath));
        ScanProjectSchemaPtr schema = ScanProjectSchemaPtr(new ScanProjectSchemaHDF5());
        
        HDF5KernelPtr hdfKernel = std::dynamic_pointer_cast<HDF5Kernel>(kernel);
        HDF5SchemaPtr hdfSchema = std::dynamic_pointer_cast<HDF5Schema>(schema);
        
        // create io object for hdf5 files
        auto scanProjectIO = std::shared_ptr<scanio::HDF5IO>(new scanio::HDF5IO(hdfKernel, hdfSchema));

        ReductionAlgorithmPtr reduction_algorithm;
        // If the user supplied valid octree reduction parameters use octree reduction otherwise use no reduction
        if (options.getOctreeVoxelSize() > 0.0f)
        {
            reduction_algorithm = std::make_shared<OctreeReductionAlgorithm>(options.getOctreeVoxelSize(), options.getOctreeMinPoints());
        }
        else
        {
            reduction_algorithm = std::make_shared<NoReductionAlgorithm>();
        }
        

        if (options.hasScanPositionIndex())
        {
            auto project = scanProjectIO->loadScanProject();
            ModelPtr model = std::make_shared<Model>();

            // Load all given scan positions
            vector<int> scanPositionIndices = options.getScanPositionIndex();
            for(int positionIndex : scanPositionIndices)
            {
                auto pos = scanProjectIO->loadScanPosition(positionIndex);
                auto lidar = pos->lidars.at(0);
                auto scan = lidar->scans.at(0);

                // std::cout << timestamp << "Loading scan position " << positionIndex << std::endl;

                // Load scan
                scan->load(reduction_algorithm);

                // std::cout << timestamp << "Scan loaded scan has " << scan->numPoints << " points" << std::endl;
                // std::cout << timestamp 
                //           << "Transforming scan: " << std::endl 
                //           <<  (project->transformation * pos->transformation * lidar->transformation * scan->transformation).cast<float>() << std::endl;

                // Transform the new pointcloud
                transformPointCloud<float>(
                    std::make_shared<Model>(scan->points),
                    (project->transformation * pos->transformation * lidar->transformation * scan->transformation).cast<float>());

                // Merge pointcloud and new scan
                // TODO: Maybe merge by allocation all needed memory first instead of constant allocations
                if (model->m_pointCloud)
                {
                    *model->m_pointCloud = mergePointBuffers(*model->m_pointCloud, *scan->points);
                }
                else
                {
                    model->m_pointCloud = std::make_shared<PointBuffer>();
                    *model->m_pointCloud = *scan->points; // Copy the first scan
                }
                scan->release();
            }
            buffer = model->m_pointCloud;
            // std::cout << timestamp << "Loaded " << buffer->numPoints() << " points" << std::endl;
        }
        else
        {    
            // === Build the PointCloud ===
            // std::cout << timestamp << "Loading scan project" << std::endl;
            ScanProjectPtr project = scanProjectIO->loadScanProject();
            
            // std::cout << project->positions.size() << std::endl;
            // The aggregated scans
            ModelPtr model = std::make_shared<Model>();
            model->m_pointCloud = nullptr;
            unsigned ctr = 0;
            for (ScanPositionPtr pos: project->positions)
            {
                // std::cout << "Loading scan position " << ctr << " / " << project->positions.size() << std::endl;
                for (LIDARPtr lidar: pos->lidars)
                {
                    for (ScanPtr scan: lidar->scans)
                    {
                        // Load scan
                        bool was_loaded = scan->loaded();
                        if (!scan->loaded())
                        {
                            scan->load();
                        }

                        // Transform the new pointcloud
                        transformPointCloud<float>(
                            std::make_shared<Model>(scan->points),
                            (project->transformation * pos->transformation * lidar->transformation * scan->transformation).cast<float>());
                        
                        // Merge pointcloud and new scan 
                        // TODO: Maybe merge by allocation all needed memory first instead of constant allocations
                        if (model->m_pointCloud)
                        {
                            *model->m_pointCloud = mergePointBuffers(*model->m_pointCloud, *scan->points);
                        }
                        else
                        {
                            model->m_pointCloud = std::make_shared<PointBuffer>();
                            *model->m_pointCloud = *scan->points; // Copy the first scan
                        }
                        
                        
                        
                        // If not previously loaded unload
                        if (!was_loaded)
                        {
                            scan->release();
                        }
                    }
                }
            }

            reduction_algorithm->setPointBuffer(model->m_pointCloud);
            buffer = reduction_algorithm->getReducedPoints();
        }

    }
    else 
    {
        buffer = model->m_pointCloud;
        std::cout << "Num Points: " << buffer->numPoints() << std::endl;
        myfile << buffer->numPoints() << ", ";
    }

    // Create a point cloud manager
    string pcm_name = options.getPCM();
    PointsetSurfacePtr<Vec> surface;

    // Create point set surface object
    if(pcm_name == "PCL")
    {
        cout << timestamp << "Using PCL as point cloud manager is not implemented yet!" << endl;
        panic_unimplemented("PCL as point cloud manager");
    }
    else if(pcm_name == "STANN" || pcm_name == "FLANN" || pcm_name == "NABO" || pcm_name == "NANOFLANN" || pcm_name == "LVR2")
    {
        
        int plane_fit_method = 0;
        
        if(options.useRansac())
        {
            plane_fit_method = 1;
        }

        // plane_fit_method
        // - 0: PCA
        // - 1: RANSAC
        // - 2: Iterative
        // TODO Delete Chrono stuff
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        surface = make_shared<AdaptiveKSearchSurface<BaseVecT>>(
            buffer,
            pcm_name,
            options.getKn(),
            options.getKi(),
            options.getKd(),
            plane_fit_method,
            options.getScanPoseFile()
        );
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

        // std::cout << "Time Building tree: " << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count() << "[ms]" << std::endl;
        myfile << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count() << ", "; 
    }
    else if(pcm_name == "LBVH_CUDA")
    {
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        surface = make_shared<CudaKSearchSurface<BaseVecT>>(
            buffer,
            options.getKn()
        );
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

        std::cout << "Time Building tree: " << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count() << "[ms]" << std::endl;
        myfile << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count() << ", ";
    }
    else
    {
        cout << timestamp << "Unable to create PointCloudManager." << endl;
        cout << timestamp << "Unknown option '" << pcm_name << "'." << endl;
        return nullptr;
    }

    // Set search options for normal estimation and distance evaluation
    surface->setKd(options.getKd());
    surface->setKi(options.getKi());
    surface->setKn(options.getKn());

    // Calculate normals if necessary
    if(!buffer->hasNormals() || options.recalcNormals())
    {
        if(options.useGPU())
        {
            #ifdef GPU_FOUND
                std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
                std::vector<float> flipPoint = options.getFlippoint();
                size_t num_points = buffer->numPoints();
                floatArr points = buffer->getPointArray();
                floatArr normals = floatArr(new float[ num_points * 3 ]);
                // std::cout << timestamp << "Generating GPU kd-tree" << std::endl;
                GpuSurface gpu_surface(points, num_points);
                

                gpu_surface.setKn(options.getKn());
                gpu_surface.setKi(options.getKi());
                gpu_surface.setFlippoint(flipPoint[0], flipPoint[1], flipPoint[2]);

                // std::cout << timestamp << "Estimating Normals GPU" << std::endl;
                gpu_surface.calculateNormals();
                gpu_surface.getNormals(normals);

                buffer->setNormalArray(normals, num_points);
                gpu_surface.freeGPU();
                std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
                myfile << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count() << ", ";
            #else
                // std::cout << timestamp << "ERROR: GPU Driver not installed" << std::endl;
                std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
                surface->calculateSurfaceNormals();
                std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

                std::cout << "Time calculating Normals: " << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count() << "[ms]" << std::endl;
                myfile << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count() << ", ";
            #endif
        }
        else
        {
            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
            surface->calculateSurfaceNormals();
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

            std::cout << "Time calculating Normals: " << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count() << "[ms]" << std::endl;
            myfile << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count() << ", ";
        }
    }
    else
    {
        cout << timestamp << "Using given normals." << endl;
    }
    if(pcm_name == "LBVH_CUDA")
    {
        // std::cout << *buffer << std::endl;
        surface = make_shared<AdaptiveKSearchSurface<BaseVecT>>(
            buffer,
            "FLANN",
            options.getKn(),
            options.getKi(),
            options.getKd(),
            0,
            options.getScanPoseFile()
        );
    }
    return surface;
}

int main(int argc, char** argv)
{
    // =======================================================================
    // Parse and print command line parameters
    // =======================================================================
    // Parse command line arguments

    // // Generates 64.000.000 points
    // lvr2::PointBufferPtr pbuffer;
    // pbuffer = lvr2::synthetic::genSpherePoints(8000,8000);

    // size_t num_points = pbuffer->numPoints();

    // lvr2::floatArr points = pbuffer->getPointArray();
    // float* points_raw = &points[0];
    
    /* 
     * argv looks like this:
     * bin/benchmark_knn_normals,
     * -p,
     * LBVH_CUDA,
     * ~/datasets/polizei/polizei30M_cut.ply
     */
    myfile.open("runtime_test.txt");

    int num_pcm = 3;
    int num_k = 2;
    int num_data = 1;

    char *pcm[] = {"LBVH_CUDA", "FLANN", "--useGPU"};                   // The tested point cloud manager
    char *k_s[] = {"5", "10"};                                          // The tested values for k
    char *data[] = {"/home/till/datasets/polizei/polizei30M_cut.ply"};  // The tested datasets

    for(int p_ = 0; p_ < num_pcm; p_++)
    {
        for(int k_ = 0; k_ < num_k; k_++)
        {
            for(int d_ = 0; d_ < num_data; d_++)
            {
                std::cout << "Testing on: " << data[d_] << std::endl;
                std::cout << "PCM: " << pcm[p_] << std::endl;
                std::cout << "K: " << k_s[k_] << std::endl;

                myfile << data[d_] << ", ";
                myfile << pcm[p_] << ", ";
                myfile << k_s[k_] << ", ";

                std::vector<char*> vec;


                int paramc = 5;

                vec.push_back("bin/benchmark_knn_normals");
                if(p_ != num_pcm - 1)
                {
                    paramc = 6;
                    vec.push_back("-p");
                }
                vec.push_back(pcm[p_]);
                vec.push_back("--kn");
                vec.push_back(k_s[k_]);
                vec.push_back(data[d_]);

                char** paramv = reinterpret_cast<char**>(&vec[0]);

                reconstruct::Options options(paramc, paramv);


                // options.printLogo();

                // Exit if options had to generate a usage message
                // (this means required parameters are missing)
                if (options.printUsage())
                {
                    return EXIT_SUCCESS;
                }

                // std::cout << options << std::endl;

                // =======================================================================
                // Load (and potentially store) point cloud
                // =======================================================================
                OpenMPConfig::setNumThreads(options.getNumThreads());

                PointsetSurfacePtr<Vec> surface;

                // Load PointCloud
                surface = loadPointCloud<Vec>(options);
                if (!surface)
                {
                    cout << "Failed to create pointcloud. Exiting." << endl;
                    exit(EXIT_FAILURE);
                }
                // ModelPtr pn(new Model(surface->pointBuffer()));
                // ModelFactory::saveModel(pn, "pointnormals.ply");

                // cout << timestamp << "Program end." << endl;

                std::cout << std::endl;
                std::cout << std::endl;
                myfile << std::endl;
            }
        }
    }

    myfile.close();

    return 0;
}
