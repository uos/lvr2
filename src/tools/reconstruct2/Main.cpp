/*
#include <lvr2/display/Arrow.hpp>
#include <lvr2/display/Color.hpp>
#include <lvr2/display/ColorMap.hpp>
#include <lvr2/display/CoordinateAxes.hpp>
#include <lvr2/display/GlTexture.hpp>
#include <lvr2/display/Grid.hpp>
#include <lvr2/display/GroundPlane.hpp>
#include <lvr2/display/InteractivePointCloud.hpp>
#include <lvr2/display/MeshCluster.hpp>
#include <lvr2/display/MultiPointCloud.hpp>
#include <lvr2/display/PointCloud.hpp>
#include <lvr2/display/PointCorrespondences.hpp>
#include <lvr2/display/PointOctree.hpp>
#include <lvr2/display/Renderable.hpp>
#include <lvr2/display/StaticMesh.hpp>
#include <lvr2/display/TexturedMesh.hpp>
#include <lvr2/display/TextureFactory.hpp>

#include <lvr2/io/ModelFactory.hpp>
#include <lvr2/io/PointBuffer.hpp>

#include <lvr2/geometry/BaseVector.hpp>
#include <lvr2/geometry/Vector.hpp>
#include <lvr2/geometry/HalfEdgeMesh.hpp>
#include <lvr2/geometry/Matrix4.hpp>
#include <lvr2/geometry/Vector.hpp>

#include <lvr2/reconstruction/AdaptiveKSearchSurface.hpp>
#include <lvr2/reconstruction/FastReconstruction.hpp>
#include <lvr2/reconstruction/PointsetGrid.hpp>
#include <lvr2/reconstruction/SharpBox.hpp>

#include <lvr2/algorithm/ReductionAlgorithms.hpp>
#include <lvr2/algorithm/NormalAlgorithms.hpp>

#include <lvr2/registration/ICPPointAlign.hpp>
#include <lvr2/registration/EigenSVDPointAlign.hpp>

#include <iostream>
#include <fstream>

using namespace lvr2;

    floatArr pts = pc->getPointArray();

    PointBufferPtr tmp1 = PointBufferPtr( new PointBuffer );
    PointBufferPtr tmp2 = PointBufferPtr( new PointBuffer );

    floatArr pts1 = floatArr( new float[3* (n/2)] );
    floatArr pts2 = floatArr( new float[3* (n - (n/2))] );
    tmp1->setPointArray(pts1, n/2);
    tmp2->setPointArray(pts2, n - (n/2));

    size_t n_2 = n / 2;

    for (size_t i = 0; i < n; i++)
    {
        if (i < n/2)
        {
            pts1[3*i + 0] = pts[i*3 + 0];
            pts1[3*i + 1] = pts[i*3 + 1];
            pts1[3*i + 2] = pts[i*3 + 2];
        }
        else
        {
            pts2[3* (i - n/2) + 0] = pts[i*3 + 0];
            pts2[3* (i - n/2) + 1] = pts[i*3 + 1];
            pts2[3* (i - n/2) + 2] = pts[i*3 + 2];
        }
    }

    ModelFactory::saveModel(ModelPtr( new Model(tmp1) ), "half1.ply");
    ModelFactory::saveModel(ModelPtr( new Model(tmp2) ), "half2.ply");
}

void test(PointBufferPtr pc_buffer)
{
    PointsetSurfacePtr<Vec> surface;

    surface  = PointsetSurfacePtr<Vec>( new AdaptiveKSearchSurface<Vec>(pc_buffer, "FLANN", 10, 10, 10, 1));


    if(!surface->pointBuffer()->hasNormals())
    {
        surface->calculateSurfaceNormals();
    }

    SharpBox<Vec>::m_surface     = surface;

    auto grid = std::make_shared<PointsetGrid<Vec, SharpBox<Vec>>>(
        10,
        surface,
        surface->getBoundingBox(),
        true,
        true
    );

    grid->calcDistanceValues();
    auto reconstruction = make_unique<FastReconstruction<Vec, SharpBox<Vec>>>(grid);

    // Create an empty mesh
    MeshBufferPtr m_buff = MeshBufferPtr( new MeshBuffer );
    HalfEdgeMesh<Vec> mesh(m_buff);
    reconstruction->getMesh(mesh);

    auto faceNormals = calcFaceNormals(mesh);

    ClusterBiMap<FaceHandle> clusterBiMap = planarClusterGrowing(mesh, faceNormals, 0.85);
    deleteSmallPlanarCluster(mesh, clusterBiMap, 10);

    ClusterPainter painter(clusterBiMap);
    auto clusterColors = DenseClusterMap<Rgb8Color>(painter.simpsons(mesh));
    auto vertexNormals = calcVertexNormals(mesh, faceNormals, *surface);

    TextureFinalizer<Vec> finalize(clusterBiMap);
    finalize.setVertexNormals(vertexNormals);
    finalize.setClusterColors(clusterColors);
    Materializer<Vec> materializer(mesh, clusterBiMap, faceNormals, *surface);
    MaterializerResult<Vec> matResult = materializer.generateMaterials();
    finalize.setMaterializerResult(matResult);
    MeshBufferPtr buffer = finalize.apply(mesh);

    ModelFactory::saveModel(ModelPtr( new Model(buffer)), "test.obj");
}

int main(int argc, char** argv)
{

    using Vec = BaseVector<float>;
    EigenSVDPointAlign<Vec> esvdpa;
    Arrow a1(1);
    Arrow a2("test");

    float c[3] = {0.0f, 0.0f, 0.0f };
    std::cout << "Color: " << c[0] << " " <<  c[1] << " " << c[2] << std::endl;
    Colors::getColor(c, Color::GREEN, ColorTable::LIGHT);
    std::cout << "Color: " << c[0] << " " <<  c[1] << " " << c[2] << std::endl;
    ModelPtr model = ModelFactory::readModel(argv[1]);

    ColorMap cm(200);

    CoordinateAxes ca1;
    CoordinateAxes ca2(3.0f);

    GlTexture glt;

    GroundPlane gp;
    GroundPlane(3, 6);

    InteractivePointCloud ipc;

    MeshCluster mc;

    std::vector<float> a;
    std::vector<float> b;

    a.push_back(3.0);
    a.push_back(2.0);
    a.push_back(1.0);
    b.push_back(6.0);
    b.push_back(4.0);
    b.push_back(4.0);

    PointCorrespondences pcorr(a, b);

    StaticMesh sm();

    TextureFactory &tf = TextureFactory::instance();
    GlTexture *gltex = tf.getTexture("../../bachelor/build/texture_17.ppm");

    if (gltex)
    {
        std::cout << "deleting texture" << std::endl;
        delete gltex;
    }

    if(model->m_pointCloud)
    {

        test(model->m_pointCloud);
        AdaptiveKSearchSurface<Vec> surf(model->m_pointCloud, "FLANN", 10, 10);
        //surf.calculateSurfaceNormals();

        ModelFactory::saveModel(model, "test.ply");

        //split_pc(model->m_pointCloud);

        std::cout << "Points: " << model->m_pointCloud->numPoints() << std::endl;
        std::cout << "Has Normals" << model->m_pointCloud->hasNormals() << std::endl;
        std::cout << "Has Colors" << model->m_pointCloud->hasColors() << std::endl;

        std::cout << "IPC test" << std::endl;
        ipc.updateBuffer(model->m_pointCloud);
        InteractivePointCloud ipc2(model->m_pointCloud);
        std::cout << "IPC test end" << std::endl;

        std::cout << "MPC test" << std::endl;
        PointCloud *pc = new PointCloud(model->m_pointCloud);
        MultiPointCloud mpc(model);
        mpc.addCloud(pc);
        std::cout << "MPC test end" << std::endl;

        std::cout << "POT test" << std::endl;
        //PointOctree(model->m_pointCloud, 2);
        std::cout << "POT test end" << std::endl;

        std::cout << "ICPPA test" << std::endl;
        Matrix4<Vec> tf;
        //tf[3] = 100.5;
        //tf[7] = 100.5;
        //tf[11] = -100.5;
        ICPPointAlign<Vec> icppa(model->m_pointCloud, model->m_pointCloud, tf);
        std::cout << icppa.match() << std::endl;
        std::cout << "ICPPA test end" << std::endl;
    }

    if(model->m_mesh)
    {
        std::cout << model->m_mesh->numVertices() << std::endl;
        std::cout << model->m_mesh->numFaces() << std::endl;
        std::cout << model->m_mesh->hasFaceColors() << std::endl;
        std::cout << model->m_mesh->hasVertexColors() << std::endl;
        std::cout << model->m_mesh->hasFaceNormals() << std::endl;
        std::cout << model->m_mesh->hasVertexNormals() << std::endl;

        std::cout << "TM test" << std::endl;
        TexturedMesh tex_mesh(model->m_mesh);
        std::cout << "TM test end" << std::endl;

        std::cout << "MC addMesh test" << std::endl;
        mc.addMesh(model->m_mesh, "test");
        std::cout << "MC addMesh test end" << std::endl;


        std::cout << "SM test" << std::endl;
        StaticMesh sm2(model);
        std::cout << "SM test end" << std::endl;

        auto textures = model->m_mesh->getTextures();

        for (auto t : textures)
        {
            t.save();
        }
    }

    std::cout << "writing model to disk" << std::endl;
    ModelFactory::saveModel(model, "mesh.obj");

    std::cout << "Matrix test" << std::endl;

    using Vec = BaseVector<float>;
    Normal<Vec> n1(1, 4, 7);
    std::cout << n1 << std::endl;
    Vector<Vec> v1(1, 2, 3);
    Matrix4<Vec> m1, m2;
    Matrix4<Vec> m3;
    m3[0] = 3;
    m3[5] = 2;
    m3[10] = 7;
    m3[15] = 9;
    std::cout << (m1 * 3) * m3 << std::endl;
    std::cout << m1 + m2 << std::endl;
    std::cout << m1 * 3 << std::endl;
    std::cout << (m1 * 3) * v1 << std::endl;
    std::cout << (m3 * 3) * n1 << std::endl;

    std::cout << "Matrix test end" << std::endl;

    return 0;

    std::cout << "creating mesh" << std::endl;
    HalfEdgeMesh<VecT> mesh(model->m_mesh);

   // mesh.debugCheckMeshIntegrity();

    auto normalMap = calcFaceNormals(mesh);
    int count = 50000000;
    auto collapsedCount = simpleMeshReduction(mesh, count, normalMap);

//    lvr2::AsciiIO io;
//    lvr2::ModelPtr model = io.read("scan.pts");

//    size_t n = model->m_pointCloud->numPoints();

//    unsigned w;
//    floatArr points = model->m_pointCloud->getPointArray();

//    std::ofstream out1("test.3d");
//    for(size_t i = 0; i < n; i++)
//    {
//        out1 << points[3 * i] << " " << points[3 * i + 1] << " " << points[3 * i + 2] << std::endl;
//    }

//    VecT offset(100, 100, 100);

//    std::ofstream out2("test1.3d");
//    FloatChannel chn = model->m_pointCloud->getFloatChannel("points");

//    for(size_t i = 0; i < n; i++)
//    {
//        chn[i] += VecT(100, 100, 100);
//        VecT d = chn[i];
//        std::cout << d << std::endl;
//        out2 << points[3 * i] << " " << points[3 * i + 1] << " " << points[3 * i + 2] << std::endl;
//    }


//    return 0;


}
//#include <lvr2/geometry/BaseVector.hpp>
//#include <lvr2/reconstruction/AdaptiveKSearchSurface.hpp>

//using namespace lvr2;

////int main(int argv, char** argc)
////{
////    AdaptiveKSearchSurface<BaseVector<float>> pss;

////    return 0;
////}
*/

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
 * approximations at the cost of running time. Increasing <i>--kd</i>
 * usually helps to generate more continuous surfaces in sparse
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

#include <lvr2/config/lvropenmp.hpp>

#include <lvr2/geometry/HalfEdgeMesh.hpp>
#include <lvr2/geometry/BaseVector.hpp>
#include <lvr2/geometry/Vector.hpp>
#include <lvr2/geometry/Normal.hpp>
#include <lvr2/attrmaps/StableVector.hpp>
#include <lvr2/attrmaps/VectorMap.hpp>
#include <lvr2/algorithm/FinalizeAlgorithms.hpp>
#include <lvr2/algorithm/NormalAlgorithms.hpp>
#include <lvr2/algorithm/ColorAlgorithms.hpp>
#include <lvr2/geometry/BoundingBox.hpp>
#include <lvr2/algorithm/Tesselator.hpp>
#include <lvr2/algorithm/ClusterPainter.hpp>
#include <lvr2/algorithm/ClusterAlgorithms.hpp>
#include <lvr2/algorithm/CleanupAlgorithms.hpp>
#include <lvr2/algorithm/ReductionAlgorithms.hpp>
#include <lvr2/algorithm/Materializer.hpp>
#include <lvr2/algorithm/Texturizer.hpp>

#include <lvr2/reconstruction/AdaptiveKSearchSurface.hpp>
#include <lvr2/reconstruction/BilinearFastBox.hpp>
#include <lvr2/reconstruction/TetraederBox.hpp>
#include <lvr2/reconstruction/FastReconstruction.hpp>
#include <lvr2/reconstruction/PointsetSurface.hpp>
#include <lvr2/reconstruction/SearchTree.hpp>
#include <lvr2/reconstruction/SearchTreeFlann.hpp>
#include <lvr2/reconstruction/HashGrid.hpp>
#include <lvr2/reconstruction/PointsetGrid.hpp>
#include <lvr2/reconstruction/SharpBox.hpp>
#include <lvr2/io/PointBuffer.hpp>
#include <lvr2/io/MeshBuffer.hpp>
#include <lvr2/io/ModelFactory.hpp>
#include <lvr2/io/PlutoMapIO.hpp>
#include <lvr2/util/Factories.hpp>
#include <lvr2/algorithm/GeometryAlgorithms.hpp>
#include <lvr2/algorithm/UtilAlgorithms.hpp>

#include <lvr2/geometry/BVH.hpp>

#include "Options.hpp"

#if defined CUDA_FOUND
    #define GPU_FOUND

    #include <lvr2/reconstruction/cuda/CudaSurface.hpp>

    typedef lvr2::CudaSurface GpuSurface;
#elif defined OPENCL_FOUND
    #define GPU_FOUND

    #include <lvr2/reconstruction/opencl/ClSurface.hpp>
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

    // Parse loaded data
    if (!model)
    {
        cout << timestamp << "IO Error: Unable to parse " << options.getInputFileName() << endl;
        return nullptr;
    }

    PointBufferPtr buffer = model->m_pointCloud;

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
                size_t num_points = buffer->numPoints();
                floatArr points = buffer->getPointArray();
                floatArr normals = floatArr(new float[ num_points * 3 ]);
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
                buffer->setNormalArray(normals, num_points);
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
        PointsetSurfacePtr<Vec> surface
    )
{
    // Determine whether to use intersections or voxelsize
    bool useVoxelsize = options.getIntersections() <= 0;
    float resolution = useVoxelsize ? options.getVoxelsize() : options.getIntersections();

    // Create a point set grid for reconstruction
    string decompositionType = options.getDecomposition();

    // Fail safe check
    if(decompositionType != "MT" && decompositionType != "MC" && decompositionType != "PMC" && decompositionType != "SF" )
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
    ClusterPainter painter(clusterBiMap);
    auto clusterColors = optional<DenseClusterMap<Rgb8Color>>(painter.simpsons(mesh));
    auto vertexColors = calcColorFromPointCloud(mesh, surface);

    // Calc normals for vertices
    auto vertexNormals = calcVertexNormals(mesh, faceNormals, *surface);

    // Prepare finalize algorithm
    TextureFinalizer<Vec> finalize(clusterBiMap);
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
    auto m = ModelPtr( new Model(buffer));

    if(options.saveOriginalData())
    {
        m->m_pointCloud = surface->pointBuffer();

        cout << "REPAIR SAVING" << endl;
    }

    cout << timestamp << "Saving mesh." << endl;
    ModelFactory::saveModel(m, "triangle_mesh.ply");
    ModelFactory::saveModel(m, "triangle_mesh.obj");
    ModelFactory::saveModel(m, "triangle_mesh.h5");

    if (matResult.m_keypoints)
    {
        // save materializer keypoints to hdf5 which is not possible with ModelFactory
        PlutoMapIO map_io("triangle_mesh.h5");
        map_io.addTextureKeypointsMap(matResult.m_keypoints.get());
    }

    cout << timestamp << "Program end." << endl;

    return 0;
}
