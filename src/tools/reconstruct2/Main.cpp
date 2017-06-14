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
#include <stdlib.h>

#include <boost/optional.hpp>


// Program options for this tool
#ifndef DEBUG
  #include "Options.hpp"
#endif


// Local includes
// #include <lvr/reconstruction/AdaptiveKSearchSurface.hpp>
// #include <lvr/reconstruction/FastReconstruction.hpp>
// #include <lvr/reconstruction/PointsetGrid.hpp>
// #include <lvr/reconstruction/FastBox.hpp>

// #include <lvr/io/PLYIO.hpp>
// #include <lvr/geometry/Matrix4.hpp>
// #include <lvr/geometry/HalfEdgeMesh.hpp>
// #include <lvr/texture/Texture.hpp>
// #include <lvr/texture/Transform.hpp>
// #include <lvr/texture/Texturizer.hpp>
// #include <lvr/texture/Statistics.hpp>
// #include <lvr/geometry/QuadricVertexCosts.hpp>
// #include <lvr/reconstruction/SharpBox.hpp>

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
#include <lvr2/util/StableVector.hpp>
#include <lvr2/util/VectorMap.hpp>

// // PCL related includes
// #ifdef LVR_USE_PCL
// #include <lvr/reconstruction/PCLKSurface.hpp>
// #endif



using boost::optional;
using std::unique_ptr;
using std::make_unique;

using lvr::timestamp;

using namespace lvr2;

using BaseVecT = BaseVector<float>;
// using PsSurface = lvr::PointsetSurface<BaseVecT>;
// using AkSurface = AdaptiveKSearchSurface<ColorVertex<float, unsigned char>, Normal<float> >;

// #ifdef LVR_USE_PCL
// typedef PCLKSurface<ColorVertex<float, unsigned char> , Normal<float> > pclSurface;
// #endif


/*
 * DUMMY TEST CODE STARTS HERE!!!
 */

void lvr2Playground()
{
    using Vec = lvr2::Vector<lvr2::BaseVector<float>>;
    using Poi = lvr2::Point<lvr2::BaseVector<float>>;

    Vec v1, v2;
    Poi p1, p2;

    v1 + p1;
    v1 + v2;
    // p1 + p2;

    v1.length();
    // v1.distance(v2);

    // p1.length();
    p1.distance(p2);

    lvr2::HalfEdgeMesh<lvr2::BaseVector<float>> mesh;

    // StableVector stuff
    lvr2::StableVector<Vec, lvr2::BaseMesh<float>::VertexHandle> vec;
    lvr2::BaseMesh<float>::VertexHandle handle1(1);
    lvr2::BaseMesh<float>::VertexHandle handle2(0);
    cout << vec.sizeUsed() << std::endl;
    vec.push_back(v1);
    cout << vec.sizeUsed() << std::endl;
    vec.push_back(v2);
    cout << vec.sizeUsed() << std::endl;
    vec.erase(handle1);
    cout << vec.sizeUsed() << std::endl;
    auto vec1 = vec[handle2];

    cout << vec.size() << std::endl;
    cout << vec1.x << std::endl;

    // VectorMap stuff 2
    cout << "VectorMap" << endl;
    lvr2::VectorMap<lvr2::BaseMesh<float>::VertexHandle, std::string> map;
    cout << map.sizeUsed() << endl;
    map.insert(handle1, "test1");
    cout << map[handle1] << std::endl;
    cout << map.sizeUsed() << endl;

    lvr2::VectorMap<lvr2::BaseMesh<float>::VertexHandle, std::string> map2(10, "test");
    for (auto i = 0; i < 10; i++) {
        lvr2::BaseMesh<float>::VertexHandle handleLoop(i);
        cout << map2[handleLoop] << endl;
    }
    cout << map2.sizeUsed() << endl;

    lvr2::BaseMesh<float>::VertexHandle handleLoop(5);
    map2[handleLoop] = "lalala";
    for (auto i = 0; i < 10; i++) {
        lvr2::BaseMesh<float>::VertexHandle handleLoop(i);
        cout << map2[handleLoop] << endl;
    }
    cout << map2.sizeUsed() << endl;

    handle1 = lvr2::BaseMesh<float>::VertexHandle(42);
    map2.insert(handle1, "42 !!");
    cout << map2.sizeUsed() << endl;
    auto opt = map2.get(handle1);
    if (opt) {
        cout << "found value! " << *opt << endl;
    }

    handle1 = lvr2::BaseMesh<float>::VertexHandle(39);
    opt = map2.get(handle1);
    if (!opt) {
        cout << "found no value!" << endl;
    }
//    map2[handle1];

    handle1 = lvr2::BaseMesh<float>::VertexHandle(42);
    map2.erase(handle1);
    cout << map2.sizeUsed() << endl;
//    cout << map2[handle1] << endl;
}

void createHouseFromNikolaus(lvr2::HalfEdgeMesh<lvr2::BaseVector<float>>& mesh)
{
    // scale
    float s = 1;

    // create house from nikolaus
    auto p0 = mesh.addVertex(BaseVector<float>(0, 0, 0));
    auto p1 = mesh.addVertex(BaseVector<float>(s, 0, 0));
    auto p2 = mesh.addVertex(BaseVector<float>(s, 0, s));
    auto p3 = mesh.addVertex(BaseVector<float>(0, 0, s));
    auto p4 = mesh.addVertex(BaseVector<float>(0, s, 0));
    auto p5 = mesh.addVertex(BaseVector<float>(s, s, 0));
    auto p6 = mesh.addVertex(BaseVector<float>(s, s, s));
    auto p7 = mesh.addVertex(BaseVector<float>(0, s, s));
    auto p8 = mesh.addVertex(BaseVector<float>(s/2, s+(s/2), s/2));

    auto bottomFace1 = mesh.addFace(p0, p1, p2);
    auto bottomFace2 = mesh.addFace(p0, p2, p3);

    auto rightFace1 = mesh.addFace(p1, p5, p6);
    auto rightFace2 = mesh.addFace(p1, p6, p2);

    auto leftFace1 = mesh.addFace(p3, p7, p4);
    auto leftFace2 = mesh.addFace(p4, p0, p3);

    auto frontFace1 = mesh.addFace(p7, p3, p2);
    auto frontFace2 = mesh.addFace(p2, p6, p7);

    auto backFace1 = mesh.addFace(p0, p4, p5);
    auto backFace2 = mesh.addFace(p5, p1, p0);

    auto roofFaceFront  = mesh.addFace(p7, p6, p8);
    auto roofFaceLeft   = mesh.addFace(p4, p7, p8);
    auto roofFaceBack   = mesh.addFace(p5, p4, p8);
    auto roofFaceRight  = mesh.addFace(p6, p5, p8);
}

void testFinalize(lvr2::HalfEdgeMesh<lvr2::BaseVector<float>>& mesh)
{
    createHouseFromNikolaus(mesh);
    mesh.debugCheckMeshIntegrity();
}

/*
 * DUMMY TEST CODE ENDS HERE!!!
 */

// optional<PsSurface::Ptr> loadPointCloud(const reconstruct::Options& options)
// {
//     // Create a point loader object
//     lvr::ModelPtr model = lvr::ModelFactory::readModel(options.getInputFileName());

//     // Parse loaded data
//     if (!model)
//     {
//         cout << timestamp << "IO Error: Unable to parse " << options.getInputFileName() << endl;
//         return boost::none;
//     }
//     lvr::PointBufferPtr p_loader = model->m_pointCloud;

//     // Create a point cloud manager
//     string pcm_name = options.getPCM();
//     PsSurface::Ptr surface;

//     // Create point set surface object
//     if(pcm_name == "PCL")
//     {
// #ifdef LVR_USE_PCL
//         surface = PsSurface::Ptr(new pclSurface(p_loader));
// #else
//         cout << timestamp << "Can't create a PCL point set surface without PCL installed." << endl;
//         return boost::none;
// #endif
//     }
//     else if(pcm_name == "STANN" || pcm_name == "FLANN" || pcm_name == "NABO" || pcm_name == "NANOFLANN")
//     {
//         akSurface* aks = new akSurface(
//                 p_loader, pcm_name,
//                 options.getKn(),
//                 options.getKi(),
//                 options.getKd(),
//                 options.useRansac(),
//                 options.getScanPoseFile()
//         );

//         surface = PsSurface::Ptr(aks);
//         // Set RANSAC flag
//         if(options.useRansac())
//         {
//             aks->useRansac(true);
//         }
//     }
//     else
//     {
//         cout << timestamp << "Unable to create PointCloudManager." << endl;
//         cout << timestamp << "Unknown option '" << pcm_name << "'." << endl;
//         cout << timestamp << "Available PCMs are: " << endl;
//         cout << timestamp << "STANN, STANN_RANSAC";
// #ifdef LVR_USE_PCL
//         cout << ", PCL";
// #endif
// #ifdef LVR_USE_NABO
//         cout << ", Nabo";
// #endif
//         cout << endl;
//         return boost::none;
//     }

//     // Set search options for normal estimation and distance evaluation
//     surface->setKd(options.getKd());
//     surface->setKi(options.getKi());
//     surface->setKn(options.getKn());

//     // Calculate normals if necessary
//     if(!surface->pointBuffer()->hasPointNormals()
//             || (surface->pointBuffer()->hasPointNormals() && options.recalcNormals()))
//     {
//         Timestamp ts;
//         surface->calculateSurfaceNormals();
//     }
//     else
//     {
//         cout << timestamp << "Using given normals." << endl;
//     }

//     // Save points and normals only
//     if(options.savePointNormals())
//     {
//         lvr::ModelPtr pn( new Model);
//         pn->m_pointCloud = surface->pointBuffer();
//         ModelFactory::saveModel(pn, "pointnormals.ply");
//     }
// }

// void setTextureOptions(const reconstruct::Options& options)
// {
//     if(options.getTexelSize())
//     {
//         Texture::m_texelSize = options.getTexelSize();
//     }

//     if(options.getTexturePack() != "")
//     {
//         Texturizer<Vertex<float> , Normal<float> >::m_filename = options.getTexturePack();
//         if(options.getStatsCoeffs())
//         {
//             float* sc = options.getStatsCoeffs();
//             for (int i = 0; i < 14; i++)
//             {
//                 Statistics::m_coeffs[i] = sc[i];
//             }
//             delete sc;
//         }
//         if(options.getNumStatsColors())
//         {
//             Texturizer<Vertex<float> , Normal<float> >::m_numStatsColors = options.getNumStatsColors();
//         }
//         if(options.getNumCCVColors())
//         {
//             Texturizer<Vertex<float> , Normal<float> >::m_numCCVColors = options.getNumCCVColors();
//         }
//         if(options.getCoherenceThreshold())
//         {
//             Texturizer<Vertex<float> , Normal<float> >::m_coherenceThreshold = options.getCoherenceThreshold();
//         }

//         if(options.getColorThreshold())
//         {
//             Texturizer<Vertex<float> , Normal<float> >::m_colorThreshold = options.getColorThreshold();
//         }
//         if(options.getStatsThreshold())
//         {
//             Texturizer<Vertex<float> , Normal<float> >::m_statsThreshold = options.getStatsThreshold();
//         }
//         if(options.getUseCrossCorr())
//         {
//             Texturizer<Vertex<float> , Normal<float> >::m_useCrossCorr = options.getUseCrossCorr();
//         }
//         if(options.getFeatureThreshold())
//         {
//             Texturizer<Vertex<float> , Normal<float> >::m_featureThreshold = options.getFeatureThreshold();
//         }
//         if(options.getPatternThreshold())
//         {
//             Texturizer<Vertex<float> , Normal<float> >::m_patternThreshold = options.getPatternThreshold();
//         }
//         if(options.doTextureAnalysis())
//         {
//             Texturizer<Vertex<float> , Normal<float> >::m_doAnalysis = true;
//         }
//         if(options.getMinimumTransformationVotes())
//         {
//             Transform::m_minimumVotes = options.getMinimumTransformationVotes();
//         }
//     }

//     if(options.getSharpFeatureThreshold())
//     {
//         SharpBox<Vertex<float> , Normal<float> >::m_theta_sharp = options.getSharpFeatureThreshold();
//     }
//     if(options.getSharpCornerThreshold())
//     {
//         SharpBox<Vertex<float> , Normal<float> >::m_phi_corner = options.getSharpCornerThreshold();
//     }
// }

int main(int argc, char** argv)
{
    try
    {
        // Parse command line arguments
        reconstruct::Options options(argc, argv);

        // Exit if options had to generate a usage message
        // (this means required parameters are missing)
        if (options.printUsage())
        {
            return EXIT_SUCCESS;
        }

        lvr::OpenMPConfig::setNumThreads(options.getNumThreads());

        std::cout << options << std::endl;

        // auto surfaceRes = loadPointCloud(options);

        // auto surface = surfaceRes ? *surfaceRes : return EXIT_FAILURE;
        // if (pcResult != 0)
        // {
        //     return EXIT_FAILURE;
        // }

        // Create an empty mesh
        lvr2::HalfEdgeMesh<lvr2::BaseVector<float>> mesh;
        testFinalize(mesh);

        // Set recursion depth for region growing
        // if(options.getDepth())
        // {
        //     mesh.setDepth(options.getDepth());
        // }

        // setTextureOptions(options);

        // Determine whether to use intersections or voxelsize
        float resolution;
        bool useVoxelsize;
        if(options.getIntersections() > 0)
        {
            resolution = options.getIntersections();
            useVoxelsize = false;
        }
        else
        {
            resolution = options.getVoxelsize();
            useVoxelsize = true;
        }

        // Create a point set grid for reconstruction
        string decomposition = options.getDecomposition();

        // Fail safe check
        if(decomposition != "MC" && decomposition != "PMC" && decomposition != "SF" )
        {
            cout << "Unsupported decomposition type " << decomposition << ". Defaulting to PMC." << endl;
            decomposition = "PMC";
        }

        // unique_ptr<GridBase> grid;
        // FastReconstructionBase<ColorVertex<float, unsigned char>, Normal<float> >* reconstruction;
        if(decomposition == "MC")
        {
            // grid = make_shared<PointsetGrid<
            //     ColorVertex<float, unsigned char>,
            //     FastBox<ColorVertex<float, unsigned char>,
            //     Normal<float>
            // >>(resolution, surface, surface->getBoundingBox(), useVoxelsize, options.extrude());
    //         PointsetGrid<ColorVertex<float, unsigned char>, FastBox<ColorVertex<float, unsigned char>, Normal<float> > >* ps_grid = static_cast<PointsetGrid<ColorVertex<float, unsigned char>, FastBox<ColorVertex<float, unsigned char>, Normal<float> > > *>(grid);
    //         ps_grid->calcDistanceValues();
    //         reconstruction = new FastReconstruction<
    //             ColorVertex<float, unsigned char>,
    //             Normal<float>,
    //             FastBox<ColorVertex<float, unsigned char>, Normal<float>>
    //         >(ps_grid);

        }
        else if(decomposition == "PMC")
        {
    //         BilinearFastBox<ColorVertex<float, unsigned char>, Normal<float> >::m_surface = surface;
    //         grid = new PointsetGrid<ColorVertex<float, unsigned char>, BilinearFastBox<ColorVertex<float, unsigned char>, Normal<float> > >(resolution, surface, surface->getBoundingBox(), useVoxelsize, options.extrude());
    //         PointsetGrid<ColorVertex<float, unsigned char>, BilinearFastBox<ColorVertex<float, unsigned char>, Normal<float> > >* ps_grid = static_cast<PointsetGrid<ColorVertex<float, unsigned char>, BilinearFastBox<ColorVertex<float, unsigned char>, Normal<float> > > *>(grid);
    //         ps_grid->calcDistanceValues();
    //         reconstruction = new FastReconstruction<
    //             ColorVertex<float, unsigned char>,
    //             Normal<float>,
    //             BilinearFastBox<ColorVertex<float, unsigned char>, Normal<float>>
    //         >(ps_grid);

        }
        else if(decomposition == "SF")
        {
    //         SharpBox<ColorVertex<float, unsigned char>, Normal<float> >::m_surface = surface;
    //         grid = new PointsetGrid<ColorVertex<float, unsigned char>, SharpBox<ColorVertex<float, unsigned char>, Normal<float> > >(resolution, surface, surface->getBoundingBox(), useVoxelsize, options.extrude());
    //         PointsetGrid<ColorVertex<float, unsigned char>, SharpBox<ColorVertex<float, unsigned char>, Normal<float> > >* ps_grid = static_cast<PointsetGrid<ColorVertex<float, unsigned char>, SharpBox<ColorVertex<float, unsigned char>, Normal<float> > > *>(grid);
    //         ps_grid->calcDistanceValues();
    //         reconstruction = new FastReconstruction<
    //             ColorVertex<float, unsigned char>,
    //             Normal<float>,
    //             SharpBox<ColorVertex<float, unsigned char>, Normal<float>>
    //         >(ps_grid);
        }



    //     // Create mesh
    //     reconstruction->getMesh(mesh);

    //     // Save grid to file
    //     if(options.saveGrid())
    //     {
    //         grid->saveGrid("fastgrid.grid");
    //     }

    //     if(options.getDanglingArtifacts())
    //     {
    //         mesh.removeDanglingArtifacts(options.getDanglingArtifacts());
    //     }

    //     // Optimize mesh
    //     mesh.cleanContours(options.getCleanContourIterations());
    //     mesh.setClassifier(options.getClassifier());
    //     mesh.getClassifier().setMinRegionSize(options.getSmallRegionThreshold());

    //     if(options.optimizePlanes())
    //     {
    //         mesh.optimizePlanes(options.getPlaneIterations(),
    //                 options.getNormalThreshold(),
    //                 options.getMinPlaneSize(),
    //                 options.getSmallRegionThreshold(),
    //                 true);

    //         mesh.fillHoles(options.getFillHoles());
    //         mesh.optimizePlaneIntersections();
    //         mesh.restorePlanes(options.getMinPlaneSize());

    //         if(options.getNumEdgeCollapses())
    //         {
    //             QuadricVertexCosts<ColorVertex<float, unsigned char> , Normal<float> > c = QuadricVertexCosts<ColorVertex<float, unsigned char> , Normal<float> >(true);
    //             mesh.reduceMeshByCollapse(options.getNumEdgeCollapses(), c);
    //         }
    //     }
    //     else if(options.clusterPlanes())
    //     {
    //         mesh.clusterRegions(options.getNormalThreshold(), options.getMinPlaneSize());
    //         mesh.fillHoles(options.getFillHoles());
    //     }

    //     // Save triangle mesh
    //     if ( options.retesselate() )
    //     {
    //         mesh.finalizeAndRetesselate(options.generateTextures(), options.getLineFusionThreshold());
    //     }
    //     else
    //     {
    //         mesh.finalize();
    //     }

    //     // Write classification to file
    //     if ( options.writeClassificationResult() )
    //     {
    //         mesh.writeClassificationResult();
    //     }

    //     // Create output model and save to file
    //     lvr::ModelPtr m( new Model( mesh.meshBuffer() ) );

    //     if(options.saveOriginalData())
    //     {
    //         m->m_pointCloud = model->m_pointCloud;
    //     }
    //     cout << timestamp << "Saving mesh." << endl;
    //     ModelFactory::saveModel( m, "triangle_mesh.ply");

    //     // Save obj model if textures were generated
    //     if(options.generateTextures())
    //     {
    //         ModelFactory::saveModel( m, "triangle_mesh.obj");
    //     }
    //     cout << timestamp << "Program end." << endl;

    }
    catch(...)
    {
        std::cout << "Unable to parse options. Call 'reconstruct --help' for more information." << std::endl;
    }
    return 0;
}
