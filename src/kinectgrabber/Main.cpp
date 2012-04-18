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

#include "io/ModelFactory.hpp"
#include "io/Timestamp.hpp"
#include "io/KinectIO.hpp"
#include "io/CoordinateTransform.hpp"
#include "Options.hpp"

#include <vector>
#include <string.h>

using namespace lssr;

int main(int argc, char** argv)
{
	// Parse command line arguments
	//kingrab::Options options(argc, argv);

	//::std::cout << options << ::std::endl;


	// Try to connect

	try
	{
		KinectIO* io = KinectIO::instance();
		int c = 0;
		vector<PointBufferPtr> scans;
		while(true && c < 20)
		{

			PointBufferPtr buffer = io->getBuffer();
			if(buffer == 0)
			{
				cout << timestamp << "No data yet..." << endl;
				// Give it some time
				usleep(100000);
			}
			else
			{

				convert(OPENGL_METERS, SLAM6D, buffer);

				//char fout[256];
				//sprintf(fout, "scan%03d.3d", c);
				//ModelFactory::saveModel(ModelPtr(new Model(buffer)), string(fout));

				//char pout[256];
				//sprintf(pout, "scan%03d.pose", c);
				//ofstream out(pout);
				//out << "0 0 0 0 0 0" << endl;

				scans.push_back(buffer);
				cout << "Buffered scan " << c << endl;
				usleep(100000);
				c++;
				// Save data
				//ModelFactory::saveModel(ModelPtr(new Model(buffer)), "pointcloud.ply");

				//			    pclSurface* s = new pclSurface( buffer, kn, kd );
				//
				//
				//
				//				// Create surface object and calculate normals
				////				akSurface* s = new akSurface(
				////						buffer, "FLANN",
				////						kn,
				////						ki,
				////						kd);
				//
				//				psSurface::Ptr surface(s);
				//
				//			    surface->setKd(kd);
				//			    surface->setKi(ki);
				//			    surface->setKn(kn);
				//				surface->calculateSurfaceNormals();
				//
				//				// Create an empty mesh and set parameters
				//				HalfEdgeMesh<cVertex, cNormal> mesh( surface );
				//				mesh.setDepth(100);
				//
				//
				//				// Create a new reconstruction object
				//				FastReconstruction<cVertex, cNormal > reconstruction(
				//						surface,
				//						voxelsize,
				//						true,
				//						"SF",
				//						true);
				//
				//				reconstruction.getMesh(mesh);
				//				mesh.setClassifier("Default");
				//				mesh.removeDanglingArtifacts(rda);
				//				mesh.optimizePlanes(planeIterations,
				//		                            planeNormalThreshold,
				//		                            minPlaneSize,
				//		                            smallRegionThreshold,
				//		                            true);
				//
				//				mesh.fillHoles(fillHoles);
				//				mesh.optimizePlaneIntersections();
				//				mesh.restorePlanes(minPlaneSize);
				//				mesh.finalize();
				//
				//				ModelPtr m( new Model( mesh.meshBuffer() ) );
				//				ModelFactory::saveModel( m, "triangle_mesh.ply" );
				//				return 0;
			}

		}

		for(size_t i = 0; i < scans.size(); i++)
		{
			char fout[256];
			sprintf(fout, "scan%03d.3d", i);
			ModelFactory::saveModel(ModelPtr(new Model(scans[i])), string(fout));
			cout << "Saving " << string(fout) << endl;
			char pout[256];

			sprintf(pout, "scan%03d.pose", i);
			ofstream out(pout);
			out << "0 0 0 0 0 0" << endl;
			out.close();
		}

	}
	catch(...)
	{
		cout << timestamp << "Kinect connection failed. Try again..." << endl;
	}



	return 0;
}
