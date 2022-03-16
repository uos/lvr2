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

#include "lvr2/reconstruction/BilinearFastBox.hpp"
#include "lvr2/reconstruction/TetraederBox.hpp"
#include "lvr2/reconstruction/FastReconstruction.hpp"
#include "lvr2/reconstruction/PointsetSurface.hpp"
#include "lvr2/reconstruction/SearchTree.hpp"
#include "lvr2/reconstruction/SearchTreeFlann.hpp"
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
#include "lvr2/io/kernels/HDF5Kernel.hpp"
#include "lvr2/io/scanio/HDF5IO.hpp"
#include "lvr2/io/scanio/ScanProjectIO.hpp"
#include "lvr2/io/schema/ScanProjectSchema.hpp"
#include "lvr2/io/schema/ScanProjectSchemaHDF5.hpp"

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
        std::shared_ptr<FeatureBuild<scanio::ScanProjectIO>> scanProjectIo = std::dynamic_pointer_cast<FeatureBuild<scanio::ScanProjectIO>>(hdf5IOPtr);

        // load scan from hdf5 file
        auto lidar = hdf5IO.LIDARIO::load(options.getScanPositionIndex(), 0);
        ScanPtr scan = lidar->scans.at(0);

        if (!scan->loaded() && scan->loadable())
        {
            scan->load();
        }
        else
        {
            std::cout << timestamp << "[Main - loadPointCloud] Unable to load points of scan " << 0 << std::endl;
        }

        buffer = scan->points;
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

    auto surface = loadPointCloud<Vec>(options);
    if (!surface)
    {
        cout << "Failed to create pointcloud. Exiting." << endl;
        return EXIT_FAILURE;
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
    lvr2::PMPMesh<Vec> mesh;

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

    Texturizer<Vec> texturizer(
        options.getTexelSize(),
        options.getTexMinClusterSize(),
        options.getTexMaxClusterSize()
    );


    std::vector<SpectralTexturizer<Vec>> spec_texters;
    // calculate how many spectral texturizers should be created
    int texturizer_count = options.getMaxSpectralChannel() - options.getMinSpectralChannel();
    texturizer_count = std::max(texturizer_count, 1);
    // initialize the SpectralTexturizers
    for(int i = 0; i < texturizer_count; i++)
    {
        SpectralTexturizer<Vec> spec_text(
            options.getTexelSize(),
            options.getTexMinClusterSize(),
            options.getTexMaxClusterSize()
        );

        spec_texters.push_back(spec_text);
    }

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
            // create hdf5 kernel and schema 
            FileKernelPtr kernel = FileKernelPtr(new HDF5Kernel(filePath));
            ScanProjectSchemaPtr schema = ScanProjectSchemaPtr(new ScanProjectSchemaHDF5());
            
            HDF5KernelPtr hdfKernel = std::dynamic_pointer_cast<HDF5Kernel>(kernel);
            HDF5SchemaPtr hdfSchema = std::dynamic_pointer_cast<HDF5Schema>(schema);
            
            // create io object for hdf5 files
            auto hdf5IO = scanio::HDF5IO(hdfKernel, hdfSchema);
            // load panorama from hdf5 file
            auto panorama = hdf5IO.HyperspectralPanoramaIO::load(options.getScanPositionIndex(), 0, 0);

            // go through all spectralTexturizers of the vector
            for(int i = 0; i < texturizer_count; i++)
            {
                // if the spectralChannel doesnt exist, skip it
                if(panorama->num_channels < options.getMinSpectralChannel() + i)
                {
                    continue;
                }
                // set the spectral texturizer with the current spectral channel
                spec_texters[i].init_image_data(panorama, std::max(options.getMinSpectralChannel(), 0) + i);
                // set the texturizer for the current spectral channel
                materializer.addTexturizer(spec_texters[i]);
            }
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
                "Mesh0",
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
                "Mesh0",
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
