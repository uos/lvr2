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

/*
 * SignalingMeshGenerator.cpp
 *
 *  Created on: 03.04.2012
 *      Author: Thomas Wiemann
 */

#include "SignalingMeshGenerator.hpp"

#include <iostream>

using std::cout;
using std::endl;

SignalingMeshGenerator::SignalingMeshGenerator()
{
	m_newData = false;
	start();
}

SignalingMeshGenerator::~SignalingMeshGenerator()
{
	// TODO Auto-generated destructor stub
}

void SignalingMeshGenerator::newPointCloud(PointBufferPtr* buffer)
{
	cout << "New Mesh" << endl;
	m_mutex.lock();
	m_pointBuffer = *buffer;
	m_newData = true;
	m_mutex.unlock();

}

void SignalingMeshGenerator::run()
{

	if(m_newData)
	{
		akSurface* s = new akSurface(
				m_pointBuffer, "FLANN",
				10,
				10,
				10);

		psSurface::Ptr surface(s);

		surface->setKd(10);
		surface->setKi(10);
		surface->setKn(10);
		surface->calculateSurfaceNormals();

		// Create an empty mesh and set parameters
		HalfEdgeMesh<cVertex, cNormal> mesh( surface );
		mesh.setDepth(100);


		// Create a new reconstruction object
		FastReconstruction<cVertex, cNormal > reconstruction(
				surface,
				0.03,
				true,
				"PMC",
				true);

		reconstruction.getMesh(mesh);
		mesh.enableRegionColoring();
		mesh.removeDanglingArtifacts(100);
		mesh.optimizePlanes(3,
				0.73,
				5,
				3,
				true);

		mesh.fillHoles(50);
		mesh.optimizePlaneIntersections();
		mesh.restorePlanes(5);
		mesh.finalize();

		m_newData = false;
	}
	else
	{
		usleep(10000);
	}
}


