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
 * ClusterTreeWidgetItem.cpp
 *
 *  Created on: 19.04.2012
 *      Author: Thomas Wiemann
 */


#include "ClusterTreeWidgetItem.h"
#include "TriangleMeshTreeWidgetItem.h"

#include "display/StaticMesh.hpp"

using lssr::StaticMesh;

void ClusterTreeWidgetItem::setRenderable(MeshCluster* c)
{

	list<StaticMesh*> meshes = c->getMeshes();
	list<StaticMesh*>::iterator it;

	for(it = meshes.begin(); it != meshes.end(); it++)
	{
		StaticMesh* m = *it;
		TriangleMeshTreeWidgetItem* item = new TriangleMeshTreeWidgetItem(TriangleMeshItem);

		// Setup supported render modes
		int modes = 0;
		size_t n_pn;
		modes |= Mesh;
		modes |= Wireframe;

		item->setName(m->Name());
		item->setNumFaces(m->getNumberOfFaces());
		item->setNumVertices(m->getNumberOfVertices());
		item->setRenderable(m);

		addChild(item);
	}

	m_renderable = c;

}


