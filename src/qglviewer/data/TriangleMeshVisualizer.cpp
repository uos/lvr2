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
 * TriangleMeshVisualizer.cpp
 *
 *  Created on: 28.03.2012
 *      Author: Thomas Wiemann
 */

#include "TriangleMeshVisualizer.hpp"

#include "display/StaticMesh.hpp"
#include "display/TexturedMesh.hpp"

#include "../widgets/TriangleMeshTreeWidgetItem.h"

TriangleMeshVisualizer::TriangleMeshVisualizer(MeshBufferPtr buffer, string name)
{
	// Test for material information
	size_t num_mat;
	buffer->getMaterialArray(num_mat);
	cout << "NUM MAT: " << num_mat << endl;
	if(!num_mat)
	{
		StaticMesh* mesh = new StaticMesh( buffer );
		m_renderable = mesh;

		TriangleMeshTreeWidgetItem* item = new TriangleMeshTreeWidgetItem(TriangleMeshItem);
		m_treeItem = item;

		int modes = 0;
		modes |= Mesh;

		if(mesh->getNormals())
		{
			modes |= VertexNormals;
		}

		item->setSupportedRenderModes(modes);
		item->setViewCentering(false);
		item->setName(name);
		item->setRenderable(mesh);
		item->setNumFaces(mesh->getNumberOfFaces());
	}
	else
	{
		TexturedMesh* mesh = new TexturedMesh( buffer);
		m_renderable = mesh;

		TriangleMeshTreeWidgetItem* item = new TriangleMeshTreeWidgetItem(TriangleMeshItem);
		m_treeItem = item;

		int modes = 0;
		modes |= Mesh;

		item->setSupportedRenderModes(modes);
		item->setViewCentering(false);
		item->setName(name);
		item->setRenderable(mesh);
		item->setNumFaces(0);
	}


}



