 /* Copyright (C) 2011 Uni Osnabrück
 * This file is part of the LAS VEGAS Reconstruction Toolkit,
 *
 * LAS VEGAS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * LAS VEGAS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA
 */


 /**
 * @mainpage LSSR Toolkit Documentation
 *
 * @section Intro Introduction
 *
 * This software delivers tools to build surface reconstructions from
 * point cloud data and a simple viewer to display the results.
 * Additionally, the found surfaces can be classified into
 * categories in terms of floor, ceiling walls etc.. The main aim of this
 * project is to deliver fast and accurate surface extraction algorithms
 * for robotic applications such as teleoperation in unknown environments
 * and localization.
 *
 * LSSR is under permanent development and runs under Linux and MacOS.
 * A Windows version will be made available soon. The software is currently
 * under heavy reorganization, so it may happen that some interfaces change.
 * Additionally, not all features have been ported to the new structure,
 * so some functionalities may not be available at all times.
 *
 * In the current version the previously available plane clustering and
 * classification algorithms are not available. Please use the previous
 * release (0.1) if your are interested in these functionalities. The
 * missing features will be re-integrated in the next release.
 *
 * @section Compiling Compiling the software
 *
 * This software uses cmake \ref [http://www.cmake.org]. The easiest way
 * to build LSSR is to perform an out of source build:
 * \verbatim
 * mkdir build
 * cd build
 * cmake .. && make
 * cd ../bin
 * \endverbatim
 *
 * External library dependencies:
 *
 * <ul>
 * <li>OpenGL</li>
 * <li>OpenGL Utility Toolkit (glut)</li>
 * <li>OpenGL Utility Library (glu)</li>
 * <li>OpenMP</li>
 * <li>Boost
 *   <ul>
 *     <li>Thread</li>
 *     <li>Filesystem</li>
 *     <li>Program Options</li>
 *     <li>System</li>
 *   </ul>
 * </li>
 * <li>Qt 4.7 or above (for viewer and qviewer)</li>
 * <li>libQGLViewer 2.3.9 or newer (for qviewer)</li>
 * <li>X.Org X11 libXi runtime library</li>
 * <li>X.Org X11 libXmu/libXmuu runtime libraries</li>
 * </ul>
 *
 *
 * @section Usage Software Usage
 *
 * LSSR comes with a tool to reconstruct surfaces from unorganized
 * points and two viewer applications. To build a surface from the
 * provided example data set, just call from the program directory
 *
 * \verbatim
 * bin/reconstruct dat/points.pts -v5
 * \endverbatim
 *
 * This command will produce a triangle mesh of the data points stored in
 * a file called "triangle_mesh.ply". The data set and the reconstruction
 * can be displayed with one of the provided viewers. Important
 * parameters for "reconstruct" are
 *
 * <table border="0">
 * <tr>
 * <td width = 10%>
 * --help
 * </td>
 * <td>
 * Prints a short description of all relevant parameters.
 * </td>
 * </tr>
 * <tr>
 * <td>-v or -i</td>
* <td>
* <p>These parameters affect the accuracy of the reconstruction.
* <i>-i</i> defines the number of intersections on the longest side
* of the scanned scene and determines the corresponding voxelsize.
* Using this parameter is useful if the scaling of a scene is
* unknown. A value of about 100 will usually generate coarse surface.
* Experiment with this value to get a tradeoff between accuracy and
* mesh size. If you know the scaling of the objects, you can set a
* fixed voxelsize by using the <i>-v</i> parameter.
* </p>
* </td>
* </tr>
* <tr>
* <td>--ki, --kn, --kd</td>
* <td>These parameters determine the number of nearest neighbors used
* for initial normal estimation (<i>--kn</i>), normal interpolation
* (<i>--ki</i>) and distance value evaluation (<i>--kd</i>). In data
* sets with a lot of noise, increasing these values can lead to better
* approximations at the cost of running time. Increasing <i>--kd</i>
* usually helps to generate more continuous surfaces in sparse
* scans, but yields in a lot of smoothing, i.e. in the
* reconstuctions, sharp features will be smoothed out.</td>
* </tr>
* </table>
*
* @section API API Description
*
* A detailed API documentation will be made available soon.
*
* @section Tutorials Tutorials
*
* A set of tutorials how to use LSSR will be made available soon.
*/

#include <iostream>
#include <memory>
#include <tuple>
#include <stdlib.h>

#include <boost/optional.hpp>


// Program options for this tool
#ifndef DEBUG
  #include "Options.hpp"
#endif


#include <lvr/config/lvropenmp.hpp>
#include <lvr/io/Timestamp.hpp>
#include <lvr/io/Model.hpp>
#include <lvr/io/ModelFactory.hpp>
#include <lvr/io/PointBuffer.hpp>
#include <lvr/reconstruction/PointsetSurface.hpp>

#include <lvr2/geometry/HalfEdgeMesh.hpp>
#include <lvr2/geometry/BaseVector.hpp>
#include <lvr2/geometry/Vector.hpp>
#include <lvr2/geometry/Point.hpp>
#include <lvr2/geometry/Normal.hpp>
#include <lvr2/attrmaps/StableVector.hpp>
#include <lvr2/attrmaps/VectorMap.hpp>
#include <lvr2/algorithm/FinalizeAlgorithm.hpp>
#include <lvr2/algorithm/NormalAlgorithms.hpp>
#include <lvr2/algorithm/ColorAlgorithms.hpp>
#include <lvr2/geometry/BoundingBox.hpp>
#include <lvr2/algorithm/Planar.hpp>
#include <lvr2/algorithm/Tesselator.hpp>
#include <lvr2/algorithm/ClusterPainter.hpp>
#include <lvr2/algorithm/ClusterAlgorithms.hpp>
#include <lvr2/algorithm/CleanupAlgorithms.hpp>
#include <lvr2/algorithm/ReductionAlgorithms.hpp>
#include <lvr2/algorithm/Materializer.hpp>
#include <lvr2/algorithm/Texturizer.hpp>

#include <lvr2/reconstruction/AdaptiveKSearchSurface.hpp>
#include <lvr2/reconstruction/BilinearFastBox.hpp>
#include <lvr2/reconstruction/FastReconstruction.hpp>
#include <lvr2/reconstruction/PointsetSurface.hpp>
#include <lvr2/reconstruction/SearchTree.hpp>
#include <lvr2/reconstruction/SearchTreeFlann.hpp>
#include <lvr2/reconstruction/HashGrid.hpp>
#include <lvr2/reconstruction/PointsetGrid.hpp>
#include <lvr2/io/PointBuffer.hpp>
#include <lvr2/io/MeshBuffer.hpp>
#include <lvr2/io/PlutoMapIO.hpp>
#include <lvr2/util/Factories.hpp>
#include <lvr2/algorithm/MeshNavAlgorithms.hpp>
#include <lvr2/algorithm/UtilAlgorithms.hpp>

#include <lvr2/geometry/BVH.hpp>

// PCL related includes
#ifdef LVR_USE_PCL
#include <lvr/reconstruction/PCLKSurface.hpp>
#endif

#if defined CUDA_FOUND
    #define GPU_FOUND

    #include <lvr/reconstruction/cuda/CudaSurface.hpp>

    typedef lvr::CudaSurface GpuSurface;
#elif defined OPENCL_FOUND
    #define GPU_FOUND

    #include <lvr/reconstruction/opencl/ClSurface.hpp>
    typedef lvr::ClSurface GpuSurface;
#endif



using boost::optional;
using std::unique_ptr;
using std::make_unique;

using lvr::timestamp;

using namespace lvr2;

using BaseVecT = BaseVector<float>;
using PsSurface = lvr::PointsetSurface<BaseVecT>;
using Vec = BaseVector<float>;


template <typename BaseVecT>
PointsetSurfacePtr<BaseVecT> loadPointCloud(const reconstruct::Options& options)
{
    // Create a point loader object
    lvr::ModelPtr model = lvr::ModelFactory::readModel(options.getInputFileName());

    // Parse loaded data
    if (!model)
    {
        cout << timestamp << "IO Error: Unable to parse " << options.getInputFileName() << endl;
        return nullptr;
    }
    auto buffer = make_shared<PointBuffer<Vec>>(*model->m_pointCloud);

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
        surface = make_shared<AdaptiveKSearchSurface<BaseVecT>>(
            buffer,
            pcm_name,
            options.getKn(),
            options.getKi(),
            options.getKd(),
            options.useRansac(),
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
                size_t num_points;
                lvr::floatArr points;
                lvr::PointBuffer old_buffer = buffer->toOldBuffer();
                points = old_buffer.getPointArray(num_points);
                lvr::floatArr normals = lvr::floatArr(new float[ num_points * 3 ]);
                std::cout << "Generate GPU kd-tree..." << std::endl;
                GpuSurface gpu_surface(points, num_points);
                std::cout << "finished." << std::endl;

                gpu_surface.setKn(options.getKn());
                gpu_surface.setKi(options.getKi());
                gpu_surface.setFlippoint(flipPoint[0], flipPoint[1], flipPoint[2]);
                std::cout << "Start Normal Calculation..." << std::endl;
                gpu_surface.calculateNormals();
                gpu_surface.getNormals(normals);
                std::cout << "finished." << std::endl;
                old_buffer.setPointNormalArray(normals, num_points);
                buffer->copyNormalsFrom(old_buffer);
                gpu_surface.freeGPU();
            #else
                std::cout << "ERROR: GPU Driver not installed" << std::endl;
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
        PointsetSurfacePtr<BaseVecT> surface
    )
{
    // Determine whether to use intersections or voxelsize
    bool useVoxelsize = options.getIntersections() <= 0;
    float resolution = useVoxelsize ? options.getVoxelsize() : options.getIntersections();

    // Create a point set grid for reconstruction
    string decompositionType = options.getDecomposition();

    // Fail safe check
    if(decompositionType != "MC" && decompositionType != "PMC" && decompositionType != "SF" )
    {
        cout << "Unsupported decomposition type " << decompositionType << ". Defaulting to PMC." << endl;
        decompositionType = "PMC";
    }

    if(decompositionType == "MC")
    {
        cout << "Decomposition type 'MC' is not implemented yet!" << endl;
        panic_unimplemented("decomposition type 'MC'");
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
    else if(decompositionType == "SF")
    {
        cout << "Decomposition type 'SF' is not implemented yet!" << endl;
        panic_unimplemented("decomposition type 'SF'");
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
    lvr::OpenMPConfig::setNumThreads(options.getNumThreads());

    auto surface = loadPointCloud<Vec>(options);
    if (!surface)
    {
        cout << "Failed to create pointcloud. Exiting." << endl;
        return EXIT_FAILURE;
    }

    // Save points and normals only
    if(options.savePointNormals())
    {
        lvr::ModelPtr pn(new lvr::Model);
        auto oldBuffer = boost::make_shared<lvr::PointBuffer>(
            surface->pointBuffer()->toOldBuffer()
        );
        pn->m_pointCloud = oldBuffer;
        lvr::ModelFactory::saveModel(pn, "pointnormals.ply");
    }


    // =======================================================================
    // Reconstruct mesh from point cloud data
    // =======================================================================
    // Create an empty mesh
    lvr2::HalfEdgeMesh<Vec> mesh;

    shared_ptr<GridBase> grid;
    unique_ptr<FastReconstructionBase<Vec>> reconstruction;
    std::tie(grid, reconstruction) = createGridAndReconstruction(options, surface);

    // Reconstruct mesh
    reconstruction->getMesh(mesh);

    // Save grid to file
    if(options.saveGrid())
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
        naiveFillSmallHoles(mesh, options.getFillHoles(), false);
    }

    // Calculate initial face normals
    auto faceNormals = calcFaceNormals(mesh);

    // Reduce mesh complexity
    const auto reductionRatio = options.getEdgeCollapseReductionRatio();
    if (reductionRatio > 0.0)
    {
        if (reductionRatio > 1.0)
        {
            throw "The reduction ratio needs to be between 0 and 1!";
        }

        // Each edge collapse removes two faces in the general case.
        // TODO: maybe we should calculate this differently...
        const auto count = static_cast<size_t>((mesh.numFaces() / 2) * reductionRatio);
        auto collapsedCount = simpleMeshReduction(mesh, count, faceNormals);
    }

    ClusterBiMap<FaceHandle> clusterBiMap;
    if(options.optimizePlanes())
    {
        clusterBiMap = iterativePlanarClusterGrowing(
            mesh,
            faceNormals,
            options.getNormalThreshold(),
            options.getPlaneIterations(),
            options.getMinPlaneSize()
        );

        if (options.getSmallRegionThreshold() > 0)
        {
            deleteSmallPlanarCluster(mesh, clusterBiMap, static_cast<size_t>(options.getSmallRegionThreshold()));
        }

        if (options.retesselate())
        {
            Tesselator<BaseVecT>::apply(mesh, clusterBiMap, faceNormals, options.getLineFusionThreshold());
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
    ClusterPainter painter(clusterBiMap);
    auto clusterColors = optional<DenseClusterMap<Rgb8Color>>(painter.simpsons(mesh));
    auto vertexColors = calcColorFromPointCloud(mesh, surface);

    // Calc normals for vertices
    auto vertexNormals = calcVertexNormals(mesh, faceNormals, *surface);

    // Prepare finalize algorithm
    ClusterFlatteningFinalizer<Vec> finalize(clusterBiMap);
    finalize.setVertexNormals(vertexNormals);

    // TODO:
    // Vielleicht sollten indv. vertex und cluster colors mit in den Materializer aufgenommen werden
    // Dafür spricht: alles mit Farben findet dann an derselben Stelle statt
    // dagegen spricht: Materializer macht aktuell nur face colors und keine vertex colors


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
    // When using textures ...
    if (options.generateTextures())
    {
        Texturizer<Vec> texturizer(
            options.getTexelSize(),
            options.getTexMinClusterSize(),
            options.getTexMaxClusterSize()
        );
        materializer.setTexturizer(texturizer);
    }
    // Generate materials
    MaterializerResult<Vec> matResult = materializer.generateMaterials();
    // When using textures ...
    if (options.generateTextures())
    {
        // Save them to disk
        materializer.saveTextures();
    }

    // Add material data to finalize algorithm
    finalize.setMaterializerResult(matResult);
    // Run finalize algorithm
    auto buffer = finalize.apply(mesh);


    // =======================================================================
    // Write all results (including the mesh) to file
    // =======================================================================
    // Create output model and save to file
    auto m = boost::make_shared<lvr::Model>(buffer->toOldBuffer(matResult));

    if(options.saveOriginalData())
    {
        m->m_pointCloud = boost::make_shared<lvr::PointBuffer>(
            surface->pointBuffer()->toOldBuffer()
        );
    }
    cout << timestamp << "Saving mesh." << endl;
    lvr::ModelFactory::saveModel(m, "triangle_mesh.h5");
    lvr::ModelFactory::saveModel(m, "triangle_mesh.ply");
    lvr::ModelFactory::saveModel(m, "triangle_mesh.obj");

    if (matResult.m_keypoints)
    {
        // save materializer keypoints to hdf5 which is not possible with lvr::ModelFactory
        lvr2::PlutoMapIO map_io("triangle_mesh.h5");
        map_io.addTextureKeypointsMap(matResult.m_keypoints.get());
    }

    cout << timestamp << "Program end." << endl;

    return 0;
}
