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
 * DataCollectorFactory.cpp
 *
 *  Created on: 07.10.2010
 *      Author: Thomas Wiemann
 */

#include "DataCollectorFactory.h"
#include "Static3DDataCollector.h"

#include "display/StaticMesh.hpp"
#include "display/PointCloud.hpp"
#include "display/MultiPointCloud.hpp"

#include "io/MeshLoader.hpp"
#include "io/PointIO.hpp"

#include "../widgets/PointCloudTreeWidgetItem.h"
#include "../widgets/TriangleMeshTreeWidgetItem.h"
#include "../widgets/MultiPointCloudTreeWidgetItem.h"

#include "io/IOFactory.hpp"

#include <boost/filesystem.hpp>
#include <boost/version.hpp>

using lssr::MeshLoader;
using lssr::PointIO;

DataCollectorFactory::DataCollectorFactory() {}

void DataCollectorFactory::create(string filename)
{
	// Get file extension
	boost::filesystem::path selectedFile(filename);

	string extension = selectedFile.extension().c_str();
	string name = selectedFile.filename().c_str();

	// Create a factory rto parse given file and extract loaders
	lssr::IOFactory io;
	io.read(filename);
	MeshLoader*   mesh_loader  = io.getMeshLoader();
	PointIO*      point_loader = io.getPointLoader();

	if(mesh_loader)
	{
	    lssr::StaticMesh* mesh = new lssr::StaticMesh(*mesh_loader);
	    TriangleMeshTreeWidgetItem* item = new TriangleMeshTreeWidgetItem(TriangleMeshItem);

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

	    Static3DDataCollector* dataCollector = new Static3DDataCollector(mesh, name, item);

	    Q_EMIT dataCollectorCreated( dataCollector );
	}

	if(point_loader)
	{
	    if(point_loader->getNumPoints() > 0)
	    {
	        // Check for multi point object
	        PointCloud* pc = new PointCloud(*point_loader);
	        PointCloudTreeWidgetItem* item = new PointCloudTreeWidgetItem(PointCloudItem);

	        // Setup supported render modes
	        int modes = 0;
	        size_t n_pn;
	        modes |= Points;
	        if(point_loader->getPointNormalArray(n_pn))
	        {
	            modes |= PointNormals;
	        }

	        item->setSupportedRenderModes(modes);
	        item->setViewCentering(false);
	        item->setName(name);
	        item->setNumPoints(pc->m_points.size());
	        item->setRenderable(pc);

	        Static3DDataCollector* dataCollector = new Static3DDataCollector(pc, name, item);
	        Q_EMIT dataCollectorCreated( dataCollector );

	    }
	}

}



