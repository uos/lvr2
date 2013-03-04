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
 * ClusterVisualizer.cpp
 *
 *  Created on: 12.04.2012
 *      Author: Thomas Wiemann
 */

#include "../widgets/ClusterTreeWidgetItem.h"

#include "ClusterVisualizer.hpp"
#include "display/MeshCluster.hpp"
#include "io/MeshBuffer.hpp"

#include <fstream>

using namespace std;
using namespace lssr;

ClusterVisualizer::ClusterVisualizer(string filename)
{
	ifstream in(filename.c_str());

	// Check!
	if(!in.good()) return;

	MeshCluster* cluster = new MeshCluster;

	// Read sub meshes
	while(in.good())
	{
		string cluster_name;
		int num_faces;
		int num_vertices;

		// Read 'header'
		in >> cluster_name;
		in >> num_faces >> num_vertices;
		BoundingBox<Vertex<float> > bb;
		// Alloc buffers
		if(num_faces && num_vertices)
		{


			floatArr vertices(new float[3 * num_vertices]);
			floatArr normals(new float[3 * num_vertices]);
			ucharArr colors(new uchar[3 * num_vertices]);
			uintArr indices(new uint[3 * num_faces]);

			// Read indices
			for(int i = 0; i < num_faces; i++)
			{
				int pos = 3 * i;
				uint a, b, c;
				in >> a >> b >> c;
				indices[pos    ] = a;
				indices[pos + 1] = b;
				indices[pos + 2] = c;
			}

			// Read vertices, normals and colors
			for(int i = 0; i < num_vertices; i++)
			{
				int pos = 3 * i;
				float x, y, z, nx, ny, nz;
				int r, g, b;

				in >> x >> y >> z >> nx >> ny >> nz >> r >> g >> b;

				vertices[pos    ] = x;
				vertices[pos + 1] = y;
				vertices[pos + 2] = z;

				normals[pos    ] = nx;
				normals[pos + 1] = ny;
				normals[pos + 2] = nz;

				colors[pos    ] = (uchar)r;
				colors[pos + 1] = (uchar)g;
				colors[pos + 2] = (uchar)b;

				bb.expand(x, y, z);
			}

			// Create Buffer
			MeshBufferPtr* buffer = new MeshBufferPtr(new MeshBuffer);
			buffer->get()->setVertexArray(vertices, num_vertices);
			buffer->get()->setVertexNormalArray(normals, num_vertices);
			buffer->get()->setVertexColorArray(colors, num_vertices);
			buffer->get()->setFaceArray(indices, num_faces);

			cluster->boundingBox()->expand(bb);
			cluster->addMesh(*buffer, cluster_name);

		}

	}

	// Create item stuff etc...
	m_renderable = cluster;

	ClusterTreeWidgetItem* item = new ClusterTreeWidgetItem(ClusterItem);
	m_treeItem = item;

	int modes = 0;
	modes |= Mesh;

	item->setSupportedRenderModes(modes);
	item->setViewCentering(false);
	item->setName("Cluster Set");
	item->setRenderable(cluster);


}


