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
#include <iostream>

#include "KinectIO.hpp"

#include "geometry/HalfEdgeMesh.hpp"
#include "reconstruction/AdaptiveKSearchSurface.hpp"
#include "reconstruction/PCLKSurface.hpp"
#include "reconstruction/FastReconstruction.hpp"
#include "io/ModelFactory.hpp"
#include "io/Timestamp.hpp"

using namespace lssr;

typedef AdaptiveKSearchSurface<cVertex, cNormal>        akSurface;
typedef PointsetSurface<cVertex>                        psSurface;
typedef PCLKSurface<cVertex, cNormal>                   pclSurface;

int main(int argc, char** argv)
{
	// Parameters
	int kn = 10;
	int ki = 10;
	int kd = 10;
	int depth = 100;
	int   planeIterations = 3;
	int	  minPlaneSize = 30;
	int   smallRegionThreshold = 0;
	int   fillHoles = 50;
	float planeNormalThreshold = 0.85;

	int  rda = 5;

	float voxelsize = 0.05;

	// Try to connect
	try
	{
		KinectIO io;
		while(true)
		{

			PointBufferPtr buffer = io.getBuffer();
			if(buffer == 0)
			{
				cout << timestamp << "No data yet..." << endl;
				// Give it some time
				usleep(100000);
			}
			else
			{
				// Save data
				ModelFactory::saveModel(ModelPtr(new Model(buffer)), "pointcloud.ply");

			    //pclSurface* s = new pclSurface( buffer, kn, kd );



				// Create surface object and calculate normals
				akSurface* s = new akSurface(
						buffer, "FLANN",
						kn,
						ki,
						kd);

				psSurface::Ptr surface(s);

			    surface->setKd(kd);
			    surface->setKi(ki);
			    surface->setKn(kn);
				surface->calculateSurfaceNormals();

				// Create an empty mesh and set parameters
				HalfEdgeMesh<cVertex, cNormal> mesh( surface );
				mesh.setDepth(100);


				// Create a new reconstruction object
				FastReconstruction<cVertex, cNormal > reconstruction(
						surface,
						voxelsize,
						true,
						"PMC",
						true);

				reconstruction.getMesh(mesh);
				mesh.enableRegionColoring();
				mesh.removeDanglingArtifacts(rda);
				mesh.optimizePlanes(planeIterations,
		                            planeNormalThreshold,
		                            minPlaneSize,
		                            smallRegionThreshold,
		                            true);
				mesh.fillHoles(fillHoles);
				mesh.optimizePlaneIntersections();
				mesh.restorePlanes(minPlaneSize);
				mesh.finalize();

				ModelPtr m( new Model( mesh.meshBuffer() ) );
				ModelFactory::saveModel( m, "triangle_mesh.ply" );
				return 0;
			}

		}
	}
	catch(...)
	{
		cout << timestamp << "Kinect connection failed. Try again..." << endl;
	}



	return 0;
}
