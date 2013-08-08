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


// Program options for this tool
#include "Options.hpp"

// Local includes
#include "reconstruction/AdaptiveKSearchSurface.hpp"
#include "reconstruction/FastReconstruction.hpp"
#include "io/PLYIO.hpp"
#include "geometry/Matrix4.hpp"
#include "geometry/HalfEdgeMesh.hpp"
#include "geometry/QuadricVertexCosts.hpp"
#include "reconstruction/SharpBox.hpp"
#include "texture/Texture.hpp"

// PCL related includes
#ifdef _USE_PCL_
#include "reconstruction/PCLKSurface.hpp"
#endif


#include <iostream>

using namespace lvr;


typedef ColorVertex<float, unsigned char>               cVertex;
typedef Normal<float>                                   cNormal;
typedef PointsetSurface<cVertex>                        psSurface;
typedef AdaptiveKSearchSurface<cVertex, cNormal>        akSurface;

#ifdef _USE_PCL_
typedef PCLKSurface<cVertex, cNormal>                   pclSurface;
#endif

/**
 * @brief   Main entry point for the LSSR surface executable
 */
int main(int argc, char** argv)
{
	try
	{
		// Parse command line arguments
		meshopt::Options options(argc, argv);

		// Exit if options had to generate a usage message
		// (this means required parameters are missing)
		if ( options.printUsage() )
		{
			return 0;
		}

		::std::cout << options << ::std::endl;


		// Create a point loader object
		ModelFactory io_factory;
		ModelPtr model = io_factory.readModel( options.getInputFileName() );

		MeshBufferPtr mesh_buffer;

		// Parse loaded data
		if ( !model )
		{
			cout << timestamp << "IO Error: Unable to parse " << options.getInputFileName() << endl;
			exit(-1);
		}
		mesh_buffer = model->m_mesh;

		if(!mesh_buffer)
		{
		    cout << timestamp << "Given file contains no supported mesh information" << endl;
		}

		// Create an empty mesh
		HalfEdgeMesh<cVertex, cNormal> mesh( mesh_buffer );

		// Set recursion depth for region growing
		if(options.getDepth())
		{
			mesh.setDepth(options.getDepth());
		}



		if(options.getDanglingArtifacts())
		{
			mesh.removeDanglingArtifacts(options.getDanglingArtifacts());
		}
		// Optimize mesh




		mesh.cleanContours(options.getCleanContourIterations());


		mesh.setClassifier(options.getClassifier());


		if(options.optimizePlanes())
		{
			mesh.optimizePlanes(options.getPlaneIterations(),
					options.getNormalThreshold(),
					options.getMinPlaneSize(),
					options.getSmallRegionThreshold(),
					true);

			mesh.fillHoles(options.getFillHoles());
			mesh.optimizePlaneIntersections();
			mesh.restorePlanes(options.getMinPlaneSize());

		}
		else
		{
			mesh.clusterRegions(options.getNormalThreshold(), options.getMinPlaneSize());
			mesh.fillHoles(options.getFillHoles());
		}

		// Save triangle mesh
		if ( options.retesselate() )
		{
			mesh.finalizeAndRetesselate(false, // Textures not yet supported in this tool
					options.getLineFusionThreshold());
		}
		else
		{
			mesh.finalize();
		}

		// Create output model and save to file
		ModelPtr m( new Model( mesh.meshBuffer() ) );


		ModelFactory::saveModel( m, "optimized_mesh.ply");


		cout << timestamp << "Program end." << endl;
	}
	catch(...)
	{
		std::cout << "Unable to parse options. Call 'reconstruct --help' for more information." << std::endl;
	}
	return 0;
}

