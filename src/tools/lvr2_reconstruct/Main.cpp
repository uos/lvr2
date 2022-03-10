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

#include <boost/optional.hpp>
#include <boost/shared_array.hpp>
#include <boost/smart_ptr/make_shared_array.hpp>

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
#include "lvr2/algorithm/SpectralTexturizer.hpp"
#include "lvr2/algorithm/RaycastingTexturizer.hpp"

#include "lvr2/reconstruction/BilinearFastBox.hpp"
#include "lvr2/reconstruction/TetraederBox.hpp"
#include "lvr2/reconstruction/FastReconstruction.hpp"
#include "lvr2/reconstruction/PointsetSurface.hpp"
#include "lvr2/reconstruction/SearchTree.hpp"
#include "lvr2/reconstruction/SearchTreeFlann.hpp"
#include "lvr2/reconstruction/HashGrid.hpp"
#include "lvr2/reconstruction/PointsetGrid.hpp"
#include "lvr2/reconstruction/SharpBox.hpp"
#include "lvr2/io/PointBuffer.hpp"
#include "lvr2/io/MeshBuffer.hpp"
#include "lvr2/io/ModelFactory.hpp"
#include "lvr2/io/PlutoMapIO.hpp"
#include "lvr2/io/meshio/HDF5IO.hpp"
#include "lvr2/io/meshio/DirectoryIO.hpp"
#include "lvr2/util/Factories.hpp"
#include "lvr2/algorithm/GeometryAlgorithms.hpp"
#include "lvr2/algorithm/UtilAlgorithms.hpp"

#include "lvr2/io/scanio/HDF5Kernel.hpp"
#include "lvr2/io/scanio/HDF5IO.hpp"
#include "lvr2/io/scanio/ScanProjectIO.hpp"
#include "lvr2/io/scanio/ScanProjectSchema.hpp"
#include "lvr2/io/scanio/ScanProjectSchemaHDF5.hpp"

#include "lvr2/geometry/BVH.hpp"

#include "lvr2/reconstruction/DMCReconstruction.hpp"

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
        cout << "loading h5 scanproject from " << filePath << endl;

        // create hdf5 kernel and schema 
        FileKernelPtr kernel = FileKernelPtr(new HDF5Kernel(filePath));
        ScanProjectSchemaPtr schema = ScanProjectSchemaPtr(new ScanProjectSchemaHDF5());
        
        HDF5KernelPtr hdfKernel = std::dynamic_pointer_cast<HDF5Kernel>(kernel);
        HDF5SchemaPtr hdfSchema = std::dynamic_pointer_cast<HDF5Schema>(schema);
        
        // create io object for hdf5 files
        auto hdf5IO = scanio::HDF5IO(hdfKernel, hdfSchema);

        auto hdf5IOPtr = std::shared_ptr<scanio::HDF5IO>(new scanio::HDF5IO(hdfKernel, hdfSchema));
        std::shared_ptr<FeatureBuild<ScanProjectIO>> scanProjectIO = std::dynamic_pointer_cast<FeatureBuild<ScanProjectIO>>(hdf5IOPtr);

        if (options.hasScanPositionIndex())
        {
            auto project = scanProjectIO->loadScanProject();
            auto pos = scanProjectIO->loadScanPosition(options.getScanPositionIndex());
            auto lidar = pos->lidars.at(0);
            auto scan = lidar->scans.at(0); 

            // Load scan
            scan->load();
            ModelPtr model = std::make_shared<Model>();
            model->m_pointCloud = scan->points;
            scan->release();

            // Transform pointcloud
            if (options.transformScanPosition())
            {
                transformPointCloud<float>(
                    model,
                    (project->transformation * pos->transformation * lidar->transformation * scan->transformation).cast<float>()
                    );
            }
            buffer = model->m_pointCloud;
        }
        else
        {    
            // === Build the PointCloud ===
            ScanProjectPtr project = scanProjectIO->loadScanProject();
            // The aggregated scans
            ModelPtr model = std::make_shared<Model>();
            model->m_pointCloud = std::make_shared<PointBuffer>();

            for (ScanPositionPtr pos: project->positions)
            {
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
                        size_t npoints_old = model->m_pointCloud->numPoints();

                        // Transform the new pointcloud
                        transformPointCloud<float>(
                            std::make_shared<Model>(scan->points),
                            (project->transformation * pos->transformation * lidar->transformation * scan->transformation).cast<float>());
                        
                        // Merge pointcloud and new scan 
                        // TODO: Maybe merge by allocation all needed memory first instead of constant allocations
                        *model->m_pointCloud = mergePointBuffers(*model->m_pointCloud, *scan->points);
                        
                        // If not previously loaded unload
                        if (!was_loaded)
                        {
                            scan->release();
                        }
                    }
                }
            }

            buffer = model->m_pointCloud;
        }

    }
    else {
        buffer = model->m_pointCloud;
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
    else if(pcm_name == "STANN" || pcm_name == "FLANN" || pcm_name == "NABO" || pcm_name == "NANOFLANN")
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

        surface = make_shared<AdaptiveKSearchSurface<BaseVecT>>(
            buffer,
            pcm_name,
            options.getKn(),
            options.getKi(),
            options.getKd(),
            plane_fit_method,
            options.getScanPoseFile()
        );
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
                std::vector<float> flipPoint = options.getFlippoint();
                size_t num_points = buffer->numPoints();
                floatArr points = buffer->getPointArray();
                floatArr normals = floatArr(new float[ num_points * 3 ]);
                std::cout << timestamp << "Generating GPU kd-tree" << std::endl;
                GpuSurface gpu_surface(points, num_points);
                

                gpu_surface.setKn(options.getKn());
                gpu_surface.setKi(options.getKi());
                gpu_surface.setFlippoint(flipPoint[0], flipPoint[1], flipPoint[2]);

                std::cout << timestamp << "Estimating Normals GPU" << std::endl;
                gpu_surface.calculateNormals();
                gpu_surface.getNormals(normals);

                buffer->setNormalArray(normals, num_points);
                gpu_surface.freeGPU();
            #else
                std::cout << timestamp << "ERROR: GPU Driver not installed" << std::endl;
                surface->calculateSurfaceNormals();
            #endif
        }
        else
        {
            surface->calculateSurfaceNormals();
        }
    }
    else
    {
        cout << timestamp << "Using given normals." << endl;
    }

    return surface;
}

std::pair<shared_ptr<GridBase>, unique_ptr<FastReconstructionBase<Vec>>>
    createGridAndReconstruction(
        const reconstruct::Options& options,
        PointsetSurfacePtr<Vec> surface
    )
{
    // Determine whether to use intersections or voxelsize
    bool useVoxelsize = options.getIntersections() <= 0;
    float resolution = useVoxelsize ? options.getVoxelsize() : options.getIntersections();

    // Create a point set grid for reconstruction
    string decompositionType = options.getDecomposition();

    // Fail safe check
    if(decompositionType != "MT" && decompositionType != "MC" && decompositionType != "DMC" && decompositionType != "PMC" && decompositionType != "SF" )
    {
        cout << "Unsupported decomposition type " << decompositionType << ". Defaulting to PMC." << endl;
        decompositionType = "PMC";
    }

    if(decompositionType == "MC")
    {
        auto grid = std::make_shared<PointsetGrid<Vec, FastBox<Vec>>>(
            resolution,
            surface,
            surface->getBoundingBox(),
            useVoxelsize,
            options.extrude()
        );
        grid->calcDistanceValues();
        auto reconstruction = make_unique<FastReconstruction<Vec, FastBox<Vec>>>(grid);
        return make_pair(grid, std::move(reconstruction));
    }
    else if(decompositionType == "PMC")
    {
        BilinearFastBox<Vec>::m_surface = surface;
        auto grid = std::make_shared<PointsetGrid<Vec, BilinearFastBox<Vec>>>(
            resolution,
            surface,
            surface->getBoundingBox(),
            useVoxelsize,
            options.extrude()
        );
        grid->calcDistanceValues();
        auto reconstruction = make_unique<FastReconstruction<Vec, BilinearFastBox<Vec>>>(grid);
        return make_pair(grid, std::move(reconstruction));
    }
    // else if(decompositionType == "DMC")
    // {
    //     auto reconstruction = make_unique<DMCReconstruction<Vec, FastBox<Vec>>>(
    //         surface,
    //         surface->getBoundingBox(),
    //         options.extrude()
    //     );
    //     return make_pair(nullptr, std::move(reconstruction));
    // }
    else if(decompositionType == "MT")
    {
        auto grid = std::make_shared<PointsetGrid<Vec, TetraederBox<Vec>>>(
            resolution,
            surface,
            surface->getBoundingBox(),
            useVoxelsize,
            options.extrude()
        );
        grid->calcDistanceValues();
        auto reconstruction = make_unique<FastReconstruction<Vec, TetraederBox<Vec>>>(grid);
        return make_pair(grid, std::move(reconstruction));
    }
    else if(decompositionType == "SF")
    {
        SharpBox<Vec>::m_surface = surface;
        auto grid = std::make_shared<PointsetGrid<Vec, SharpBox<Vec>>>(
            resolution,
            surface,
            surface->getBoundingBox(),
            useVoxelsize,
            options.extrude()
        );
        grid->calcDistanceValues();
        auto reconstruction = make_unique<FastReconstruction<Vec, SharpBox<Vec>>>(grid);
        return make_pair(grid, std::move(reconstruction));
    }

    return make_pair(nullptr, nullptr);
}

template <typename Vec>
void addSpectralTexturizers(const reconstruct::Options& options, lvr2::Materializer<Vec>& materializer)
{
    if (!options.hasScanPositionIndex())
    {
        return;
    }

    // TODO: Check if the scanproject has spectral data
    boost::filesystem::path selectedFile( options.getInputFileName());
    std::string filePath = selectedFile.generic_path().string();

    // create hdf5 kernel and schema 
    HDF5KernelPtr hdfKernel = std::make_shared<HDF5Kernel>(filePath);
    HDF5SchemaPtr hdfSchema = std::make_shared<ScanProjectSchemaHDF5>();
    
    // create io object for hdf5 files
    auto hdf5IO = scanio::HDF5IO(hdfKernel, hdfSchema);
    // load panorama from hdf5 file
    auto panorama = hdf5IO.HyperspectralPanoramaIO::load(options.getScanPositionIndex(), 0, 0);

    // If there is no spectral data
    if (!panorama)
    {
        return;
    }

    int texturizer_count = options.getMaxSpectralChannel() - options.getMinSpectralChannel();
    texturizer_count = std::max(texturizer_count, 0);

    // go through all spectralTexturizers of the vector
    for(int i = 0; i < texturizer_count; i++)
    {
        // if the spectralChannel doesnt exist, skip it
        if(panorama->num_channels < options.getMinSpectralChannel() + i)
        {
            continue;
        }

        auto spec_text = std::make_shared<SpectralTexturizer<Vec>>(
            options.getTexelSize(),
            options.getTexMinClusterSize(),
            options.getTexMaxClusterSize()
        );

        // set the spectral texturizer with the current spectral channel
        spec_text->init_image_data(panorama, std::max(options.getMinSpectralChannel(), 0) + i);
        // set the texturizer for the current spectral channel
        materializer.addTexturizer(spec_text);
    }
}

template <typename MeshVec, typename ClusterVec>
void addRGBTexturizer(const reconstruct::Options& options, lvr2::Materializer<Vec>& materializer, const BaseMesh<MeshVec>& mesh, const ClusterBiMap<ClusterVec> clusters)
{
    boost::filesystem::path selectedFile( options.getInputFileName());
    std::string extension = selectedFile.extension().string();
    std::string filePath = selectedFile.generic_path().string();

    if (extension != ".h5")
    {
        std::cout << timestamp << "Cannot add RGB Texturizer because the scanproject is not a HDF5 file" << std::endl;
        return;
    }

    HDF5KernelPtr kernel = std::make_shared<HDF5Kernel>(filePath);
    HDF5SchemaPtr schema = std::make_shared<ScanProjectSchemaHDF5>();
    
    // create io object for hdf5 files
    auto hdf5IO = scanio::HDF5IO(kernel, schema);

    ScanProjectPtr project = hdf5IO.loadScanProject();

    auto texturizer = std::make_shared<RaycastingTexturizer<Vec>>(
        options.getTexelSize(),
        options.getTexMinClusterSize(),
        options.getTexMaxClusterSize(),
        mesh,
        clusters,
        project
    );

    materializer.addTexturizer(texturizer);
}

template <typename BaseMeshT>
auto loadAndReconstructMesh(reconstruct::Options options)
{
    auto surface = loadPointCloud<Vec>(options);
    if (!surface)
    {
        cout << "Failed to create pointcloud. Exiting." << endl;
        exit(EXIT_FAILURE);
    }

    // Save points and normals only
    if(options.savePointNormals())
    {
        ModelPtr pn(new Model(surface->pointBuffer()));
        ModelFactory::saveModel(pn, "pointnormals.ply");
    }

    // =======================================================================
    // Reconstruct mesh from point cloud data
    // =======================================================================
    // Create an empty mesh
    BaseMeshT mesh;

    shared_ptr<GridBase> grid;
    unique_ptr<FastReconstructionBase<Vec>> reconstruction;
    std::tie(grid, reconstruction) = createGridAndReconstruction(options, surface);

    // Reconstruct mesh
    reconstruction->getMesh(mesh);

    // Save grid to file
    if(options.saveGrid() && grid)
    {
        grid->saveGrid("fastgrid.grid");
    }

    // =======================================================================
    // Optimize mesh
    // =======================================================================
    if(options.getDanglingArtifacts())
    {
        cout << timestamp << "Removing dangling artifacts" << endl;
        removeDanglingCluster(mesh, static_cast<size_t>(options.getDanglingArtifacts()));
    }

    // Magic number from lvr1 `cleanContours`...
    cleanContours(mesh, options.getCleanContourIterations(), 0.0001);

    // Fill small holes if requested
    if(options.getFillHoles())
    {
        mesh.fillHoles(options.getFillHoles());
    }

    // Reduce mesh complexity
    const auto reductionRatio = options.getEdgeCollapseReductionRatio();
    if (reductionRatio > 0.0)
    {
        if (reductionRatio > 1.0)
        {
            throw "The reduction ratio needs to be between 0 and 1!";
        }

        size_t old = mesh.numVertices();
        size_t target = old * (1.0 - reductionRatio);
        std::cout << timestamp << "Trying to remove " << old - target << " / " << old << " vertices." << std::endl;
        mesh.simplify(target);
        std::cout << timestamp << "Removed " << old - mesh.numVertices() << " vertices." << std::endl;
    }

    return std::make_tuple(std::move(mesh), surface);
}

template <typename BaseMeshT>
auto loadExistingMesh(reconstruct::Options options)
{
    meshio::HDF5IO io(
        std::make_shared<HDF5Kernel>(options.getInputMeshFile()),
        std::make_shared<MeshSchemaHDF5>()
    );

    std::cout << timestamp << "Loading existing mesh '" << options.getInputMeshName() << "' from file '" << options.getInputFileName() << "'" << std::endl;

    BaseMeshT mesh(io.loadMesh(options.getInputMeshName()));
    
    PointBufferPtr buffer = std::make_shared<PointBuffer>();
    floatArr points(new float[3]);
    buffer->setPointArray(points, 1);
    PointsetSurfacePtr<Vec> surface = std::make_shared<AdaptiveKSearchSurface<Vec>>(
            buffer,
            options.getPCM(),
            options.getKn(),
            options.getKi(),
            options.getKd(),
            0,
            options.getScanPoseFile());

    

    return std::make_tuple(std::move(mesh), surface);
}

int main(int argc, char** argv)
{
    // =======================================================================
    // Parse and print command line parameters
    // =======================================================================
    // Parse command line arguments
    reconstruct::Options options(argc, argv);

    options.printLogo();

    // Exit if options had to generate a usage message
    // (this means required parameters are missing)
    if (options.printUsage())
    {
        return EXIT_SUCCESS;
    }

    std::cout << options << std::endl;

    // =======================================================================
    // Load (and potentially store) point cloud
    // =======================================================================
    OpenMPConfig::setNumThreads(options.getNumThreads());

    lvr2::PMPMesh<Vec> mesh;
    PointsetSurfacePtr<Vec> surface;

    if (options.useExistingMesh())
    {
        std::tie(mesh, surface) = loadExistingMesh<lvr2::PMPMesh<Vec>>(options);
    }
    else
    {
        std::tie(mesh, surface) = loadAndReconstructMesh<lvr2::PMPMesh<Vec>>(options);
    }

    // Calculate face normals
    auto faceNormals = calcFaceNormals(mesh);

    ClusterBiMap<FaceHandle> clusterBiMap;
    if(options.optimizePlanes())
    {
        clusterBiMap = iterativePlanarClusterGrowingRANSAC(
            mesh,
            faceNormals,
            options.getNormalThreshold(),
            options.getPlaneIterations(),
            options.getMinPlaneSize()
        );

        if(options.getSmallRegionThreshold() > 0)
        {
            deleteSmallPlanarCluster(mesh, clusterBiMap, static_cast<size_t>(options.getSmallRegionThreshold()));
        }

        if(options.getFillHoles())
        {
            mesh.fillHoles(options.getFillHoles());
        }

        cleanContours(mesh, options.getCleanContourIterations(), 0.0001);

        if (options.retesselate())
        {
            Tesselator<Vec>::apply(mesh, clusterBiMap, faceNormals, options.getLineFusionThreshold());
        }
    }
    else
    {
        clusterBiMap = planarClusterGrowing(mesh, faceNormals, options.getNormalThreshold());
    }

    // =======================================================================
    // Finalize mesh
    // =======================================================================
    // Prepare color data for finalizing

    ColorGradient::GradientType t = ColorGradient::gradientFromString(options.getClassifier());

    ClusterPainter painter(clusterBiMap);
    auto clusterColors = boost::optional<DenseClusterMap<RGB8Color>>(painter.colorize(mesh, t));
    auto vertexColors = calcColorFromPointCloud(mesh, surface);

    // Calc normals for vertices
    auto vertexNormals = calcVertexNormals(mesh, faceNormals, *surface);

    // Prepare finalize algorithm
    TextureFinalizer<Vec> finalize(clusterBiMap);
    finalize.setVertexNormals(vertexNormals);

    // Vertex colors:
    // If vertex colors should be generated from pointcloud:
    if (options.vertexColorsFromPointcloud())
    {
        // set vertex color data from pointcloud
        finalize.setVertexColors(*vertexColors);
    }
    else if (clusterColors)
    {
        // else: use simpsons painter for vertex coloring
        finalize.setClusterColors(*clusterColors);
    }

    // Materializer for face materials (colors and/or textures)
    Materializer<Vec> materializer(
        mesh,
        clusterBiMap,
        faceNormals,
        *surface
    );

    auto texturizer = std::make_shared<Texturizer<Vec>>(
        options.getTexelSize(),
        options.getTexMinClusterSize(),
        options.getTexMaxClusterSize()
    );




    // When using textures ...
    if (options.generateTextures())
    {

        boost::filesystem::path selectedFile( options.getInputFileName());
        std::string filePath = selectedFile.generic_path().string();

        if(selectedFile.extension().string() != ".h5") {
            materializer.setTexturizer(texturizer);
        } 
        else 
        {
            addSpectralTexturizers(options, materializer);

            addRGBTexturizer(options, materializer, mesh, clusterBiMap);
        }
    }

    // Generate materials
    MaterializerResult<Vec> matResult = materializer.generateMaterials();

    // Add material data to finalize algorithm
    finalize.setMaterializerResult(matResult);
    // Run finalize algorithm
    auto buffer = finalize.apply(mesh);

    // When using textures ...
    if (options.generateTextures())
    {
        // Set optioins to save them to disk
        //materializer.saveTextures();
        buffer->addIntAtomic(1, "mesh_save_textures");
        buffer->addIntAtomic(1, "mesh_texture_image_extension");
    }

    // =======================================================================
    // Write all results (including the mesh) to file
    // =======================================================================
    // Create output model and save to file
    auto m = ModelPtr( new Model(buffer));

    if(options.saveOriginalData())
    {
        m->m_pointCloud = surface->pointBuffer();

        cout << "REPAIR SAVING" << endl;
    }

    for(const std::string& output_filename : options.getOutputFileNames())
    {
        boost::filesystem::path selectedFile( output_filename );
        std::string extension = selectedFile.extension().string();
        cout << timestamp << "Saving mesh to "<< output_filename << "." << endl;

        if (extension == ".h5")
        {
            /* TODO: TESTING IO move this to a part of this program where it makes sense*/

            std::cout << timestamp << "[Experimental] Saving using MeshIO" << std::endl;

            HDF5KernelPtr kernel = HDF5KernelPtr(new HDF5Kernel(output_filename));
            MeshSchemaHDF5Ptr schema = MeshSchemaHDF5Ptr(new MeshSchemaHDF5());
            auto mesh_io = meshio::HDF5IO(kernel, schema);

            mesh_io.saveMesh(
                options.getMeshName(),
                buffer
                );

            continue;
        }

        if (extension == "")
        {
            /* TODO: TESTING IO move this to a part of this program where it makes sense*/

            std::cout << timestamp << "[Experimental] Saving using MeshIO" << std::endl;

            DirectoryKernelPtr kernel = DirectoryKernelPtr(new DirectoryKernel(output_filename));
            MeshSchemaDirectoryPtr schema = MeshSchemaDirectoryPtr(new MeshSchemaDirectory());
            auto mesh_io = meshio::DirectoryIO(kernel, schema);

            mesh_io.saveMesh(
                options.getMeshName(),
                buffer
                );

            continue;
        }

        ModelFactory::saveModel(m, output_filename);
    }

    if (matResult.m_keypoints)
    {
        // save materializer keypoints to hdf5 which is not possible with ModelFactory
        //PlutoMapIO map_io("triangle_mesh.h5");
        //map_io.addTextureKeypointsMap(matResult.m_keypoints.get());
    }

    cout << timestamp << "Program end." << endl;

    return 0;
}
