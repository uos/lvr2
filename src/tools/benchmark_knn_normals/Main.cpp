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
#include <math.h>

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

#include "lvr2/registration/ReductionAlgorithm.hpp"

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
#include "lvr2/reconstruction/cuda/LBVHIndex.hpp"

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

// Globals
const char *path = "/home/tests/runtime_tests/";
std::ofstream myfile(path);

long int g_build_time = 0;
long int g_knn_time = 0;
size_t g_num_points;

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
PointsetSurfacePtr<BaseVecT> loadPointCloud(const benchmark::Options& options)
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
        // myfile << buffer->numPoints() << ", ";
        g_num_points = buffer->numPoints();
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
        // myfile << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count() << ", "; 
        g_build_time += std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count();
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
        // myfile << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count() << ", ";
        g_build_time += std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count();
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
                // myfile << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count() << ", ";
                g_knn_time += std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count();
            #else
                // std::cout << timestamp << "ERROR: GPU Driver not installed" << std::endl;
                std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
                surface->calculateSurfaceNormals();
                std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

                std::cout << "Time calculating Normals: " << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count() << "[ms]" << std::endl;
                // myfile << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count() << ", ";
                g_knn_time += std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count();
            #endif
        }
        else
        {
            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
            surface->calculateSurfaceNormals();
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

            std::cout << "Time calculating Normals: " << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count() << "[ms]" << std::endl;
            // myfile << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count() << ", ";
            g_knn_time += std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count();
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

void saveCubeCloud(int num_1, int num_2, std::string filename)
{
    size_t num_points = num_1 * num_2 * 6;

    float* points = (float*) malloc(sizeof(float) * num_points * 3);

    // Set Bounding Box of the cube
    float min_x = 0.0f;
    float min_y = 0.0f;
    float min_z = 0.0f;

    float max_x = 1.0f;
    float max_y = 1.0f;
    float max_z = 1.0f;

    // Create faces of the cube

    // First face at max x value
    int offset = 0;

    for(int first = 0; first < num_1; first++)
    {
        for(int second = 0; second < num_2; second++)
        {
            points[3 * num_1 * num_2 * offset + 3 * first * num_2 + 3 * second + 0] = max_x - (max_x / num_1);             // x coord
            points[3 * num_1 * num_2 * offset + 3 * first * num_2 + 3 * second + 1] = (min_y + first) / num_1;     // y coord
            points[3 * num_1 * num_2 * offset + 3 * first * num_2 + 3 * second + 2] = (min_z + second) / num_2;    // z coord
        }
    }

    // Second face at min x value
    offset = 1;

    for(int first = 0; first < num_1; first++)
    {
        for(int second = 0; second < num_2; second++)
        {
            points[3 * num_1 * num_2 * offset + 3 * first * num_2 + 3 * second + 0] = min_x;     // x coord
            points[3 * num_1 * num_2 * offset + 3 * first * num_2 + 3 * second + 1] = (min_y + first) / num_1; // y coord
            points[3 * num_1 * num_2 * offset + 3 * first * num_2 + 3 * second + 2] = (min_z + second) / num_2; // z coord
        }
    }

    // third face at max y value
    offset = 2;

    for(int first = 0; first < num_1; first++)
    {
        for(int second = 0; second < num_2; second++)
        {
            points[3 * num_1 * num_2 * offset + 3 * first * num_2 + 3 * second + 0] = (min_x + first) / num_1;     // x coord
            points[3 * num_1 * num_2 * offset + 3 * first * num_2 + 3 * second + 1] = max_y - (max_y / num_1); // y coord
            points[3 * num_1 * num_2 * offset + 3 * first * num_2 + 3 * second + 2] = (min_z + second) / num_2; // z coord
        }
    }

    // fourth face at min y value
    offset = 3;

    for(int first = 0; first < num_1; first++)
    {
        for(int second = 0; second < num_2; second++)
        {
            points[3 * num_1 * num_2 * offset + 3 * first * num_2 + 3 * second + 0] = (min_x + first) / num_1;     // x coord
            points[3 * num_1 * num_2 * offset + 3 * first * num_2 + 3 * second + 1] = min_y; // y coord
            points[3 * num_1 * num_2 * offset + 3 * first * num_2 + 3 * second + 2] = (min_z + second) / num_2; // z coord
        }
    }

    // fifth face at max z value
    offset = 4;

    for(int first = 0; first < num_1; first++)
    {
        for(int second = 0; second < num_2; second++)
        {
            points[3 * num_1 * num_2 * offset + 3 * first * num_2 + 3 * second + 0] = (min_x + first) / num_1;     // x coord
            points[3 * num_1 * num_2 * offset + 3 * first * num_2 + 3 * second + 1] = (min_y + second) / num_2; // y coord
            points[3 * num_1 * num_2 * offset + 3 * first * num_2 + 3 * second + 2] = max_z - (max_z / num_1); // z coord
        }
    }

    // sixth face at min z value
    offset = 5;

    for(int first = 0; first < num_1; first++)
    {
        for(int second = 0; second < num_2; second++)
        {
            points[3 * num_1 * num_2 * offset + 3 * first * num_2 + 3 * second + 0] = (min_x + first) / num_1;     // x coord
            points[3 * num_1 * num_2 * offset + 3 * first * num_2 + 3 * second + 1] = (min_y + second) / num_2; // y coord
            points[3 * num_1 * num_2 * offset + 3 * first * num_2 + 3 * second + 2] = min_z; // z coord
        }
    }

    PointBufferPtr pbuffer(new PointBuffer());

    pbuffer->setPointArray(floatArr(&points[0]), num_points);

    ModelPtr cubeModel(new Model(pbuffer));

    ModelFactory::saveModel(cubeModel, filename);
}

void calculateNormalsCube(std::string filename, float* normals, bool save_model=false)
{
    ModelPtr model = ModelFactory::readModel(filename);

    // Get the points
    PointBufferPtr pbuffer = model->m_pointCloud;
    size_t num_points = model->m_pointCloud->numPoints();

    lvr2::floatArr points = pbuffer->getPointArray();
    float* points_raw = &points[0];

    // Get the Bounding Box of the cube
    float max_x = 0;
    float max_y = 0;
    float max_z = 0;
    float min_x = FLT_MAX;
    float min_y = FLT_MAX;
    float min_z = FLT_MAX;

    for(int i = 0; i < num_points; i++)
    {
        float x = points_raw[3 * i + 0];
        if(x > max_x)
        {
            max_x = x; 
        }
        if(x < min_x)
        {
            min_x = x;
        }
        float y = points_raw[3 * i + 1];
        if(y > max_y)
        {
            max_y = y; 
        }
        if(y < min_y)
        {
            min_y = y;
        }
        float z = points_raw[3 * i + 2];
        if(z > max_z)
        {
            max_z = z; 
        }
        if(z < min_z)
        {
            min_z = z;
        }
    }

    // Reset normals
    for(int i = 0; i < num_points; i++)
    {
        normals[3 * i + 0] = 0.0f;
        normals[3 * i + 1] = 0.0f;
        normals[3 * i + 2] = 0.0f;
    }

    for(int i = 0; i < num_points; i++)
    {
        // Set normals on the faces
        float x = points_raw[3 * i + 0];
        float y = points_raw[3 * i + 1];
        float z = points_raw[3 * i + 2];

        if(x == max_x)
        {
            normals[3 * i + 0] = 1.0f;
        }
        if(y == max_y)
        {
            normals[3 * i + 1] = 1.0f;
        }
        if(z == max_z)
        {
            normals[3 * i + 2] = 1.0f;
        }

        if(x == min_x)
        {
            normals[3 * i + 0] = -1.0f;
        }
        if(y == min_y)
        {
            normals[3 * i + 1] = -1.0f;
        }
        if(z == min_z)
        {
            normals[3 * i + 2] = -1.0f;
        }
    }

    float mag;
   
    for(int i = 0; i < num_points; i++)
    {
        // Normalize the normals
        mag = sqrt( normals[3 * i + 0] * normals[3 * i + 0] +
                    normals[3 * i + 1] * normals[3 * i + 1] +
                    normals[3 * i + 2] * normals[3 * i + 2]);

        normals[3 * i + 0] = normals[3 * i + 0] / mag;
        normals[3 * i + 1] = normals[3 * i + 1] / mag;
        normals[3 * i + 2] = normals[3 * i + 2] / mag;

        // Flip normals if necessary
        float flip_x = 10000000.0f;
        float flip_y = 10000000.0f;
        float flip_z = 10000000.0f;

        float vertex_x = points_raw[3 * i + 0];
        float vertex_y = points_raw[3 * i + 1];
        float vertex_z = points_raw[3 * i + 2];

        float x_dir = flip_x - vertex_x;
        float y_dir = flip_y - vertex_y;
        float z_dir = flip_z - vertex_z;

        float scalar =  x_dir * normals[3 * i + 0] + 
                        y_dir * normals[3 * i + 1] + 
                        z_dir * normals[3 * i + 2];

        if(scalar < 0)
        {
            normals[3 * i + 0] = -normals[3 * i + 0];
            normals[3 * i + 1] = -normals[3 * i + 1];
            normals[3 * i + 2] = -normals[3 * i + 2];
        }
    }

    if(save_model) 
    {
        pbuffer->setNormalArray(floatArr(&normals[0]), num_points);
        ModelPtr sphereModel(new Model(pbuffer));

        ModelFactory::saveModel(sphereModel, "cubeNormals.ply");
    }
}

void saveSphereCloud(int num_long, int num_lat, std::string filename)
{
    lvr2::PointBufferPtr pbuffer;
    pbuffer = lvr2::synthetic::genSpherePoints(num_long, num_lat);

    size_t num_points = pbuffer->numPoints();

    lvr2::floatArr points = pbuffer->getPointArray();
    float* points_raw = &points[0];

    ModelPtr sphereModel(new Model(pbuffer));
   
    ModelFactory::saveModel(sphereModel, filename);
}

void calculateNormalsSphere(std::string filename, float* normals, bool save_model=false)
{
    ModelPtr model = ModelFactory::readModel(filename);

    // Get the points
    PointBufferPtr pbuffer = model->m_pointCloud;
    size_t num_points = model->m_pointCloud->numPoints();

    lvr2::floatArr points = pbuffer->getPointArray();
    float* points_raw = &points[0];

    float mag;
   
    for(int i = 0; i < num_points; i++)
    {
        mag = sqrt( points_raw[3 * i + 0] * points_raw[3 * i + 0] +
                    points_raw[3 * i + 1] * points_raw[3 * i + 1] +
                    points_raw[3 * i + 2] * points_raw[3 * i + 2]);

        normals[3 * i + 0] = points_raw[3 * i + 0] / mag;
        normals[3 * i + 1] = points_raw[3 * i + 1] / mag;
        normals[3 * i + 2] = points_raw[3 * i + 2] / mag;

        // Flip normals if necessary
        float flip_x = 10000000.0f;
        float flip_y = 10000000.0f;
        float flip_z = 10000000.0f;

        float vertex_x = points_raw[3 * i + 0];
        float vertex_y = points_raw[3 * i + 1];
        float vertex_z = points_raw[3 * i + 2];

        float x_dir = flip_x - vertex_x;
        float y_dir = flip_y - vertex_y;
        float z_dir = flip_z - vertex_z;

        float scalar =  x_dir * normals[3 * i + 0] + 
                        y_dir * normals[3 * i + 1] + 
                        z_dir * normals[3 * i + 2];

        if(scalar < 0)
        {
            normals[3 * i + 0] = -normals[3 * i + 0];
            normals[3 * i + 1] = -normals[3 * i + 1];
            normals[3 * i + 2] = -normals[3 * i + 2];
        }
    }

    if(save_model) 
    {
        pbuffer->setNormalArray(floatArr(&normals[0]), num_points);
        ModelPtr sphereModel(new Model(pbuffer));

        ModelFactory::saveModel(sphereModel, "sphereNormals.ply");
    }
}

float getVectorAngle(const float* normals, const float* true_normals, int idx_1, int idx_2)
{
    // get scalar
    float scalar =  normals[3 * idx_1 + 0] * true_normals[3 * idx_2 + 0] +
                    normals[3 * idx_1 + 1] * true_normals[3 * idx_2 + 1] +
                    normals[3 * idx_1 + 2] * true_normals[3 * idx_2 + 2];

    // get the magnitudes
    float mag_1 = sqrt( normals[3 * idx_1 + 0] * normals[3 * idx_1 + 0] +
                        normals[3 * idx_1 + 1] * normals[3 * idx_1 + 1] +
                        normals[3 * idx_1 + 2] * normals[3 * idx_1 + 2]);

    float mag_2 = sqrt( true_normals[3 * idx_2 + 0] * true_normals[3 * idx_2 + 0] +
                        true_normals[3 * idx_2 + 1] * true_normals[3 * idx_2 + 1] +
                        true_normals[3 * idx_2 + 2] * true_normals[3 * idx_2 + 2]);

    // Ignore uninitialised normals
    if(mag_1 == 0.0f || mag_2 == 0.0f)
    {
        return 0.0f;
    }

    // --useGPU produces nan values in some normals. These normals will be skipped.
    if(!(mag_1 < 0 || mag_1 >= 0))
    {
        return 0.0f;
    }
    if(!(mag_2 < 0 || mag_2 >= 0))
    {
        return 0.0f;
    }

    // Get the angle in radian
    float value = scalar / (mag_1 * mag_2);
    float angle_rad;

    // Make acos() safe - somehow acos() has a problem with normalized vectors
    if (value <= -1.0f) 
    {
        angle_rad = M_PI;
    } 
    else if (value>=1.0) 
    {
        angle_rad = 0;
    } 
    else 
    {
        angle_rad = acos(value);
    }
    
    // Convert angle to degrees
    float angle = angle_rad * 180 / M_PI;

    // Check if angle is nan
    if(!(angle > 0 || angle <= 0))
    {
        std::cout << "Index: " << idx_1 << std::endl;
        std::cout << "Normal: (" << normals[3 * idx_1 + 0] << ", " << normals[3 * idx_1 + 1] << "," << normals[3 * idx_1 + 2]  << ")" << std::endl;
        std::cout << "True normal: (" << true_normals[3 * idx_2 + 0] << ", " << true_normals[3 * idx_2 + 1] << "," << true_normals[3 * idx_2 + 2]  << ")" << std::endl;
        std::cout << "Mag 1: " << mag_1 << std::endl;
        std::cout << "Mag 2: " << mag_2 << std::endl;
        std::cout << "Val: " << value << std::endl;
        std::cout << "Radian: " << angle_rad << std::endl;
        std::cout << "Angle: " << angle << std::endl;
        exit(0);

    }

    return abs(angle);
}

void getNormalDifference(float* normals, float* true_normals, size_t num_normals, float* diff, float* avg, float* max, float* min, int* num_180, float* variance)
{
    

    for(int i = 0; i < num_normals; i++)
    {
        float d = getVectorAngle(normals, true_normals, i, i);
        if(d > *max)
        {
            *max = d;
        }
        if(d < *min)
        {
            *min = d;
        }

        diff[i] = d;
        *avg += d / num_normals;

        if(d >= 180.0f)
        {
            *num_180 += 1;
        }
    }

    // Calculate variance
    float var = 0.0f;
    for(int i = 0; i < num_normals; i++)
    {
        var += ((diff[i] - *avg) * (diff[i] - *avg)) / num_normals;
    }
    *variance = var;

    return;
}

int main(int argc, char** argv)
{

    // float normals[] = {0.0, 1.0, 0.0};
    // float true_normals[] = {1.0,1.0,0.0};

    // std::cout << "Angle: " << getVectorAngle(normals, true_normals, 0, 0) << std::endl;
    // exit(0);

    // saveCubeCloud(1000, 1000, "cube6M.ply");
    // float* normals = (float*) malloc(sizeof(float) * 1000 * 1000 * 6 * 3);
    // calculateNormalsCube("cube6M.ply", normals, true);
    // exit(0);
    // saveSphereCloud(200, 200, "sphere40K.ply");


    // // ################ Test Normal Quality #######################################################
    // myfile.open("normal_quality_test_2.csv");

    // int num_pcm = 3;
    // int num_k = 5;
    // int num_data = 4;

    // char *pcm[] = {"LBVH_CUDA", "FLANN", "--useGPU"};                   // The tested point cloud manager                     
    // // char *pcm[] = {"--useGPU"};
    // // char *pcm[] = {"FLANN"};
    // char *k_s[] = {"10", "25", "50", "75", "100"};                            // The tested values for k  
    // // char *k_s[] = {"75"}; 
    // char *data[] = {                                                    // The tested datasets
    //     "/home/tstueckemann/datasets/synthetic/sphere40K.ply",
    //     "/home/tstueckemann/datasets/synthetic/sphere4M.ply",           // ###########################################
    //     "/home/tstueckemann/datasets/synthetic/cube60K.ply",            // Different calc method for cube and sphere
    //     "/home/tstueckemann/datasets/synthetic/cube6M.ply"              // ###########################################
    // };    
    // myfile << "n,pcm,k,avg,max,min,num180" << std::endl;                                                    

    // for(int p_ = 0; p_ < num_pcm; p_++)
    // {
    //     for(int k_ = 0; k_ < num_k; k_++)
    //     {
    //         for(int d_ = 0; d_ < num_data; d_++)
    //         {
    //             std::cout << "Testing on: " << data[d_] << std::endl;
    //             std::cout << "PCM: " << pcm[p_] << std::endl;
    //             std::cout << "K: " << k_s[k_] << std::endl;

    //             std::vector<char*> vec;

    //             int paramc = 5;

    //             vec.push_back("bin/benchmark_knn_normals");
    //             // TODO Comment in when using --GPU
    //             if(p_ != num_pcm - 1)
    //             {
    //                 paramc = 6;
    //                 vec.push_back("-p");
    //             }
    //             // paramc = 6;
    //             // vec.push_back("-p");
    //             vec.push_back(pcm[p_]);
    //             vec.push_back("--kn");
    //             vec.push_back(k_s[k_]);
    //             vec.push_back(data[d_]);

    //             char** paramv = reinterpret_cast<char**>(&vec[0]);

    //             benchmark::Options options(paramc, paramv);

    //             // Exit if options had to generate a usage message
    //             // (this means required parameters are missing)
    //             if (options.printUsage())
    //             {
    //                 return EXIT_SUCCESS;
    //             }

    //             // std::cout << options << std::endl;

    //             // =======================================================================
    //             // Load (and potentially store) point cloud
    //             // =======================================================================
    //             OpenMPConfig::setNumThreads(options.getNumThreads());

    //             PointsetSurfacePtr<Vec> surface;

    //             // Load PointCloud
    //             surface = loadPointCloud<Vec>(options);
    //             if (!surface)
    //             {
    //                 cout << "Failed to create pointcloud. Exiting." << endl;
    //                 exit(EXIT_FAILURE);
    //             }

    //             // Get the calculated normals
    //             PointBufferPtr pb = surface->pointBuffer();
    //             size_t num_normals = pb->numPoints();
    //             floatArr normalArr = pb->getNormalArray();
    //             float* normals = &normalArr[0];
                
    //             // FLANN doesnt seem to flip the normals. So do that here again
    //             bool equal = (pcm[p_] == "FLANN");

    //             if(equal)
    //             {
    //                 std::cout << "FLipping normals for FLANN" << std::endl;
    //                 floatArr pointArr = pb->getPointArray();

    //                 for(int i = 0; i < num_normals; i++)
    //                 {
    //                     // Flip normals if necessary
    //                     float flip_x = 10000000.0f;
    //                     float flip_y = 10000000.0f;
    //                     float flip_z = 10000000.0f;

    //                     float vertex_x = pointArr[3 * i + 0];
    //                     float vertex_y = pointArr[3 * i + 1];
    //                     float vertex_z = pointArr[3 * i + 2];

    //                     float x_dir = flip_x - vertex_x;
    //                     float y_dir = flip_y - vertex_y;
    //                     float z_dir = flip_z - vertex_z;

    //                     float scalar =  x_dir * normals[3 * i + 0] + 
    //                                     y_dir * normals[3 * i + 1] + 
    //                                     z_dir * normals[3 * i + 2];

    //                     if(scalar < 0)
    //                     {
    //                         normals[3 * i + 0] = -normals[3 * i + 0];
    //                         normals[3 * i + 1] = -normals[3 * i + 1];
    //                         normals[3 * i + 2] = -normals[3 * i + 2];
    //                     }
    //                 }
    //             }

    //             // Get the true normals
    //             float* true_normals = (float*) malloc(sizeof(float) * 3 * num_normals);
    //             // TODO
    //             if(d_ < 2)
    //             {
    //                 calculateNormalsSphere(data[d_], true_normals);
    //             }
    //             else{
    //                 calculateNormalsCube(data[d_], true_normals);
    //             }

    //             float* diff = (float*) malloc(sizeof(float) * num_normals);

    //             float avg = 0.0f;
    //             float max = 0.0f;
    //             float min = FLT_MAX;
    //             int num_180 = 0;
    //             getNormalDifference(normals, true_normals, num_normals, diff, &avg, &max, &min, &num_180);

    //             std::cout << "Average normal difference: " << avg << " degree" << std::endl;

    //             myfile  << num_normals << "," << pcm[p_] << "," << k_s[k_] << "," 
    //                     << avg << "," << max << "," << min << "," << num_180 << ",";
                
    //         }
    //     }
    // }
    // myfile.close();

    // ############################################################################################
    //                  TESTING SMTH
    // #################### Create Normal Histogram CSV ###########################################
    myfile.open("normal_quality_perc_5bin.csv");

    int num_pcm = 3;
    int num_k = 5;
    int num_data = 4;

    char *pcm[] = {"LBVH_CUDA", "FLANN", "--useGPU"};                   // The tested point cloud manager                     
    // char *pcm[] = {"--useGPU"};
    // char *pcm[] = {"FLANN"};
    char *k_s[] = {"10", "25", "50", "75", "100"};                            // The tested values for k  
    // char *k_s[] = {"75"}; 
    char *data[] = {                                                    // The tested datasets
        "/home/tstueckemann/datasets/synthetic/sphere40K.ply",
        "/home/tstueckemann/datasets/synthetic/sphere4M.ply",           // ###########################################
        "/home/tstueckemann/datasets/synthetic/cube60K.ply",            // Different calc method for cube and sphere
        "/home/tstueckemann/datasets/synthetic/cube6M.ply"              // ###########################################
    };    
    myfile << "n,pcm,k,avg,bin,num_norm,perc_norm,var," << std::endl;                                                 

    for(int p_ = 0; p_ < num_pcm; p_++)
    {
        for(int k_ = 0; k_ < num_k; k_++)
        {
            for(int d_ = 0; d_ < num_data; d_++)
            {
                std::cout << "Testing on: " << data[d_] << std::endl;
                std::cout << "PCM: " << pcm[p_] << std::endl;
                std::cout << "K: " << k_s[k_] << std::endl;

                std::vector<char*> vec;

                int paramc = 5;

                vec.push_back("bin/benchmark_knn_normals");
                // TODO Comment in when using --GPU
                if(p_ != num_pcm - 1)
                {
                    paramc = 6;
                    vec.push_back("-p");
                }
                // paramc = 6;
                // vec.push_back("-p");
                vec.push_back(pcm[p_]);
                vec.push_back("--kn");
                vec.push_back(k_s[k_]);
                vec.push_back(data[d_]);

                char** paramv = reinterpret_cast<char**>(&vec[0]);

                benchmark::Options options(paramc, paramv);

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

                // Get the calculated normals
                PointBufferPtr pb = surface->pointBuffer();
                size_t num_normals = pb->numPoints();
                floatArr normalArr = pb->getNormalArray();
                float* normals = &normalArr[0];
                
                // FLANN doesnt seem to flip the normals. So do that here again
                bool equal = (pcm[p_] == "FLANN");

                
                std::cout << "FLipping normals again for all PCMs" << std::endl;
                floatArr pointArr = pb->getPointArray();

                for(int i = 0; i < num_normals; i++)
                {
                    // Flip normals if necessary
                    float flip_x = 10000000.0f;
                    float flip_y = 10000000.0f;
                    float flip_z = 10000000.0f;

                    float vertex_x = pointArr[3 * i + 0];
                    float vertex_y = pointArr[3 * i + 1];
                    float vertex_z = pointArr[3 * i + 2];

                    float x_dir = flip_x - vertex_x;
                    float y_dir = flip_y - vertex_y;
                    float z_dir = flip_z - vertex_z;

                    float scalar =  x_dir * normals[3 * i + 0] + 
                                    y_dir * normals[3 * i + 1] + 
                                    z_dir * normals[3 * i + 2];

                    if(scalar < 0)
                    {
                        normals[3 * i + 0] = -normals[3 * i + 0];
                        normals[3 * i + 1] = -normals[3 * i + 1];
                        normals[3 * i + 2] = -normals[3 * i + 2];
                    }
                }
                

                // Get the true normals
                float* true_normals = (float*) malloc(sizeof(float) * 3 * num_normals);
                // TODO
                if(d_ < 2)
                {
                    calculateNormalsSphere(data[d_], true_normals);
                }
                else{
                    calculateNormalsCube(data[d_], true_normals);
                }

                float* diff = (float*) malloc(sizeof(float) * num_normals);

                float avg = 0.0f;
                float max = 0.0f;
                float min = FLT_MAX;
                int num_180 = 0;
                float variance = 0.0f;

                getNormalDifference(normals, true_normals, num_normals, diff, &avg, &max, &min, &num_180, &variance);

                std::cout << "Average normal difference: " << avg << " degree" << std::endl;

                int step_size = 5;
                // "n,pcm,k,avg,bin,num_norm,perc_nom,"
                for(int i = 0; i < 181; i += step_size)
                {
                    myfile  << num_normals << "," << pcm[p_] << "," << k_s[k_] << "," 
                            << avg << ",";
                    myfile << i << ",";
                    
                    int c = 0;
                    for(int j = 0; j < num_normals; j++) {
                        if(diff[j] >= i && diff[j] < i+step_size)
                        {
                            c++; 
                        }
                    }
                    myfile << c << ",";

                    float perc = (float) c / (float) num_normals;
                    myfile << perc << "," << variance << "," << std::endl;
                }
                
            }
        }
    }
    myfile.close();
    

    // ############################################################################################
    // ################ Create Sphere Point Cloud #################################################
    //

    // // Generates 64.000.000 points
    // lvr2::PointBufferPtr pbuffer;
    // pbuffer = lvr2::synthetic::genSpherePoints(3000,3000);

    // size_t num_points = pbuffer->numPoints();

    // lvr2::floatArr points = pbuffer->getPointArray();
    // float* points_raw = &points[0];

    // ModelPtr sphereModel(new Model(pbuffer));

    // std::cout << "Sphere points: " << sphereModel->m_pointCloud->numPoints() << std::endl;
   
    // // Save the new model as test.ply
    // ModelFactory::saveModel(sphereModel, "sphere.ply");

    


    // ######################################################################################
    /* 
     * argv looks like this:
     * bin/benchmark_knn_normals,
     * -p,
     * LBVH_CUDA,
     * ~/datasets/polizei/polizei30M_cut.ply
     */
    // #####################################################################################
    // USE THIS TO SUBSAMPLE PLY FILES
    // #####################################################################################
    // benchmark::Options opt(argc, argv);
    // ModelPtr model = ModelFactory::readModel(opt.getInputFileName());

    // // Get the points
    // PointBufferPtr pbuffer = model->m_pointCloud;
    // size_t num_points = model->m_pointCloud->numPoints();

    // std::cout << "Before: " << num_points << std::endl;

    // float voxelSize = 0.03f;
    // size_t maxPointsperVoxel = 240;

    // OctreeReduction ocRed(pbuffer, voxelSize, maxPointsperVoxel);
    // PointBufferPtr redPBuffer;

    // redPBuffer = ocRed.getReducedPoints();
    // // floatArr arr = redPBuffer->getFloatArray();

    // ModelPtr redModel(new Model(redPBuffer));
    // // model->m_mesh.reset();
    // // model->m_pointCloud.reset();

    // // model->m_pointCloud->setNormalArray(*redPBuffer);

    // std::cout << "After: " << redModel->m_pointCloud->numPoints() << std::endl;
   
    // // Save the new model as test.ply
    // ModelFactory::saveModel(redModel, "test.ply");


    // exit(0);
    // ############################################################################################
    // ################### Test combined kernel ########################################################
    // myfile.open("max_k_test_10M.csv");

    // int epochs = 1;
    // int num_pcm = 1;
    // int num_k = 3;
    // int num_data = 1;

    // // char *pcm[] = {"LBVH_CUDA", "FLANN", "--useGPU"};                   // The tested point cloud manager
    // char *pcm[] = {"LBVH_CUDA"};
    // // char *k_s[] = {"4", "8", "16", "32", "64", "96", "128"};                  // The tested values for k     
    // char *k_s[] = {"200", "500", "1000"};                                              
    // char *data[] = {                                                    // The tested datasets
    //     // "/home/tstueckemann/datasets/polizei/polizei1M.ply",
    //     // "/home/tstueckemann/datasets/polizei/polizei2M.ply",
    //     // "/home/tstueckemann/datasets/polizei/polizei5M.ply",
    //     "/home/tstueckemann/datasets/polizei/polizei10M.ply"
    //     // "/home/tstueckemann/datasets/polizei/polizei20M.ply",
    //     // "/home/tstueckemann/datasets/polizei/polizei30M.ply"
    //     // "/home/tstueckemann/datasets/polizei/polizei15M.ply",
    //     // "/home/tstueckemann/datasets/polizei/polizei25M.ply"
    // };                                                                  

    // for(int p_ = 0; p_ < num_pcm; p_++)
    // {
    //     for(int k_ = 0; k_ < num_k; k_++)
    //     {
    //         for(int d_ = 0; d_ < num_data; d_++)
    //         {
    //             std::cout << "Testing on: " << data[d_] << std::endl;
    //             std::cout << "PCM: " << pcm[p_] << std::endl;
    //             std::cout << "K: " << k_s[k_] << std::endl;

    //             std::vector<char*> vec;

    //             int paramc = 5;

    //             vec.push_back("bin/benchmark_knn_normals");
    //             // TODO Comment in when using --GPU
    //             // if(p_ != num_pcm - 1)
    //             // {
    //             //     paramc = 6;
    //             //     vec.push_back("-p");
    //             // }
    //             paramc = 6;
    //             vec.push_back("-p");
    //             vec.push_back(pcm[p_]);
    //             vec.push_back("--kn");
    //             vec.push_back(k_s[k_]);
    //             vec.push_back(data[d_]);

    //             char** paramv = reinterpret_cast<char**>(&vec[0]);

    //             benchmark::Options options(paramc, paramv);


    //             // options.printLogo();

    //             // Exit if options had to generate a usage message
    //             // (this means required parameters are missing)
    //             if (options.printUsage())
    //             {
    //                 return EXIT_SUCCESS;
    //             }

    //             // std::cout << options << std::endl;

    //             // =======================================================================
    //             // Load (and potentially store) point cloud
    //             // =======================================================================
    //             OpenMPConfig::setNumThreads(options.getNumThreads());

    //             // TODO Do more than one epoch?
    //             for(int e_ = 0; e_ < epochs; e_++)
    //             {
    //                 PointsetSurfacePtr<Vec> surface;

    //                 // Load PointCloud
    //                 surface = loadPointCloud<Vec>(options);
    //                 if (!surface)
    //                 {
    //                     cout << "Failed to create pointcloud. Exiting." << endl;
    //                     exit(EXIT_FAILURE);
    //                 }

    //             }
    //             // ModelPtr pn(new Model(surface->pointBuffer()));
    //             // ModelFactory::saveModel(pn, "pointnormals.ply");

    //             // cout << timestamp << "Program end." << endl;

    //             // Output file:
    //             // (dataset,) numPoints, PCM, K, buildTime, knnTime
    //             // myfile << data[d_] << ", ";
    //             myfile << g_num_points << ", ";
    //             myfile << pcm[p_] << ", ";
    //             myfile << k_s[k_] << ", ";
    //             myfile << g_build_time / epochs << ", ";
    //             myfile << g_knn_time / epochs << ", ";
    //             myfile << std::endl;

    //             g_build_time = 0;
    //             g_knn_time = 0;

    //             std::cout << std::endl;
    //             std::cout << std::endl;
    //         }
    //     }
    // }

    // myfile.close();
    // ############################################################

    // #################### Test separate kernels #########################

    // long int build_time = 0;
    // long int knn_time = 0;
    // long int normals_time = 0;

    // myfile.open("runtime_test.txt");
    // myfile << "n,pcm,k,build,knn,normals," << std::endl;

    // int k_s[] = {4, 8, 16, 32, 64, 96, 128};
    // char *data[] = {                                                    // The tested datasets
    //     // "/home/tstueckemann/datasets/polizei/polizei1M.ply"
    //     // "/home/tstueckemann/datasets/polizei/polizei2M.ply"
    //     // "/home/tstueckemann/datasets/polizei/polizei5M.ply"
    //     "/home/tstueckemann/datasets/polizei/polizei10M.ply"
    //     // "/home/tstueckemann/datasets/polizei/polizei20M.ply"
    //     // "/home/tstueckemann/datasets/polizei/polizei30M.ply"
    //     // "/home/tstueckemann/datasets/polizei/polizei15M.ply",
    //     // "/home/tstueckemann/datasets/polizei/polizei25M.ply"
    // };   

    // for(int k_ = 0; k_ < 7; k_++)
    // {
    //     for(int d_ = 0; d_ < 1; d_++)
    //     {            
             
    //         int paramc = 2;
    //         std::vector<char*> vec;
    //         vec.push_back("bin/benchmark_knn_normals");
    //         vec.push_back(data[d_]);

    //         char** paramv = reinterpret_cast<char**>(&vec[0]);    

    //         benchmark::Options opt(paramc, paramv);

    //         ModelPtr model = ModelFactory::readModel(opt.getInputFileName());

    //         // Get the points
    //         PointBufferPtr pbuffer = model->m_pointCloud;
    //         size_t num_points = model->m_pointCloud->numPoints();
    //         floatArr points = pbuffer->getPointArray();
    //         float* points_raw = &points[0];
    //         int K = k_s[k_];

    //         myfile << num_points << ",";
    //         myfile << "SEPARATE" << ",";
    //         myfile << K << ",";

            
    //         std::cout << "Testing on n=" << num_points << ", k=" << K << std::endl;

    //         lvr2::lbvh::LBVHIndex lbvh(32, true, true);

    //         std::chrono::steady_clock::time_point begin_build = std::chrono::steady_clock::now();
    //         lbvh.build(points_raw, num_points);
    //         std::chrono::steady_clock::time_point end_build = std::chrono::steady_clock::now();

    //         build_time = std::chrono::duration_cast<std::chrono::milliseconds> (end_build - begin_build).count();
    //         std::cout << "Time building LBVH: " << build_time << "[ms]" << std::endl;
    //         myfile << build_time << ",";


    //         size_t size =  3 * num_points;

    //         // Get the queries
    //         size_t num_queries = num_points;

    //         float* queries = points_raw;

    //         // Create the normal array
    //         float* normals = (float*) malloc(sizeof(float) * num_queries * 3);

    //         // Create the return arrays
    //         unsigned int* n_neighbors_out;
    //         unsigned int* indices_out;
    //         float* distances_out;

    //         // Malloc the output arrays here
    //         n_neighbors_out = (unsigned int*) malloc(sizeof(unsigned int) * num_queries);
    //         indices_out = (unsigned int*) malloc(sizeof(unsigned int) * num_queries * K);
    //         distances_out = (float*) malloc(sizeof(float) * num_queries * K);

    //         std::cout << "KNN Search..." << std::endl;
    //         // Process the queries 


    //         std::chrono::steady_clock::time_point begin_knn = std::chrono::steady_clock::now();
    //         lbvh.kSearch(
    //             queries, 
    //             num_queries,
    //             K,
    //             n_neighbors_out, 
    //             indices_out, 
    //             distances_out
    //         );
    //         std::chrono::steady_clock::time_point end_knn = std::chrono::steady_clock::now();
            
    //         knn_time = std::chrono::duration_cast<std::chrono::milliseconds> (end_knn - begin_knn).count();
    //         std::cout << "Time kNN Search: " << knn_time << "[ms]" << std::endl;
    //         myfile << knn_time << ",";

            
    //         std::cout << "Normals..." << std::endl;
    //         // Calculate the normals

    //         std::chrono::steady_clock::time_point begin_normals = std::chrono::steady_clock::now();
    //         lbvh.calculate_normals(
    //             normals, 
    //             num_queries,        
    //             queries, 
    //             num_queries, 
    //             K,
    //             n_neighbors_out, 
    //             indices_out
    //         );
    //         std::chrono::steady_clock::time_point end_normals = std::chrono::steady_clock::now();

    //         normals_time = std::chrono::duration_cast<std::chrono::milliseconds> (end_normals - begin_normals).count();
    //         std::cout << "Time Normal Calculation: " << normals_time << "[ms]" << std::endl;
    //         myfile << normals_time << "," << std::endl;

    //         std::cout << "Done!" << std::endl;
    //     }
    // }
    // ##########################################################################################
    
    return 0;
}
