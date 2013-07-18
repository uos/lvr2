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

#include "io/Model.hpp"
#include "display/StaticMesh.hpp"

using namespace lvr;

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

void ClusterTreeWidgetItem::saveCluster(string filename)
{
	// Open output file
	ofstream out(filename.c_str());

	// Iterate over all sub-items
    QTreeWidgetItemIterator w_it( this);
    while (*w_it)
    {
    	if( (*w_it)->type() >= TriangleMeshItem)
    	{
    		TriangleMeshTreeWidgetItem* item = static_cast<TriangleMeshTreeWidgetItem*>(*w_it);
    		if(item->isSelected())
    		{
    			// Get buffer
    			lvr::ModelPtr model = item->renderable()->model();
    			if(model)
    			{
    				lvr::MeshBufferPtr mesh = model->m_mesh;

    				if(mesh)
    				{
    					size_t nFaces;
    					size_t nVertices;
    					size_t nColors;

    					floatArr vertices = mesh->getVertexArray(nVertices);
    					ucharArr colors = mesh->getVertexColorArray(nColors);
    					uintArr indices = mesh->getFaceArray(nFaces);

    					// Write mesh info
    					out << item->renderable()->Name();
    					out << nFaces << " " << nColors << endl;
    					for(size_t i = 0; i < nFaces; i++)
    					{
    						out << indices[3 * i] << " " << indices[3 * 1 + 1] << " " << indices[3 * i + 2] << endl;
    					}

    					for(size_t i = 0; i < nVertices; i++)
    					{
    						out << vertices[3 * i] << " " << vertices[3 * 1 + 1] << " " << vertices[3 * i + 2] << " ";
    						out << (int)colors[3 * i] << " " << (int)colors[3 * 1 + 1] << " " << (int)colors[3 * i + 2] << endl;
    					}
    				}
    			}
    		}
    	}
        ++w_it;
    }

    out.close();

}


