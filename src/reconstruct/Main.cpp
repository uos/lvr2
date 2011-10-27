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

#include "Options.hpp"
#include "reconstruction/PCLPointCloudManager.hpp"
#include "reconstruction/StannPointCloudManager.hpp"
#include "reconstruction/FastReconstruction.hpp"
#include "io/PLYIO.hpp"
#include "geometry/Matrix4.hpp"
#include "geometry/TriangleMesh.hpp"
#include "geometry/HalfEdgeMesh.hpp"
#include <iostream>

using namespace lssr;

/**
 * @brief   Main entry point for the LSSR surface executable
 */
int main(int argc, char** argv)
{
    // Parse command line arguments
    reconstruct::Options options(argc, argv);

    // Exit if options had to generate a usage message
    // (this means required parameters are missing)
    if (options.printUsage()) return 0;

    ::std::cout << options << ::std::endl;


    // Create a point loader object
    size_t num_points;
    IOFactory io_factory;
    io_factory.read(options.getInputFileName());
    PointIO* p_loader = io_factory.getPointIO();

    // Create a point cloud manager
    string pcm_name = options.getPCM();
    PointCloudManager<ColorVertex<float, unsigned char>, Normal<float> >* pcm;
    if(pcm_name == "PCL")
    {
#ifdef _USE_PCL_
        cout << timestamp << "Creating PCL point cloud manager." << endl;
        pcm = new PCLPointCloudManager<ColorVertex<float, unsigned char>, Normal<float> > (p_loader);
#else
        cout << timestamp << "PCL bindings not found. Using STANN instead." << endl;
        pcm = new StannPointCloudManager<ColorVertex<float, unsigned char>, Normal<float> > (p_loader);
#endif
    }
    else
    {
        cout << timestamp << "Creating STANN point cloud manager." << endl;
        pcm = new StannPointCloudManager<ColorVertex<float, unsigned char>, Normal<float> > (p_loader);
    }

    pcm->setKD(options.getKd());
    pcm->setKI(options.getKi());
    pcm->setKN(options.getKn());
    pcm->calcNormals();

    // Create an empty mesh
    //TriangleMesh<Vertex<float>, Normal<float> > mesh;
    HalfEdgeMesh<ColorVertex<float, unsigned char>, Normal<float> > mesh(pcm);

    // Determine weather to use intersections or voxelsize
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

    // Create a new reconstruction object
    FastReconstruction<ColorVertex<float, unsigned char>, Normal<float> > reconstruction(*pcm, resolution, useVoxelsize);
    reconstruction.getMesh(mesh);

    mesh.removeDanglingArtifacts(options.getDanglingArtifacts());

    // Optimize mesh
    if(options.optimizePlanes())
    {
        if(options.colorRegions()) mesh.enableRegionColoring();
        mesh.optimizePlanes(options.getPlaneIterations(),
                            options.getNormalThreshold(),
                            options.getMinPlaneSize(),
                            options.getSmallRegionThreshold(),
                            true);

        mesh.fillHoles(options.getFillHoles());

        mesh.optimizePlaneIntersections();

        mesh.restorePlanes();

        mesh.optimizePlanes(3,
                            options.getNormalThreshold(),
                            options.getMinPlaneSize(),
                            0,
                            false );

    }

//    mesh.tester();

    // Save triangle mesh
    if(options.retesselate())
	 {
		 mesh.finalizeAndRetesselate(options.generateTextures());
	 } else
	 {
		 mesh.finalize();
	 }
    mesh.save("triangle_mesh.ply");
    mesh.saveObj("triangle_mesh.obj");


    cout << timestamp << "Program end." << endl;

	return 0;
}

