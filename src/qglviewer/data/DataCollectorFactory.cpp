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

#include "display/Grid.hpp"
#include "display/StaticMesh.hpp"
#include "display/PointCloud.hpp"
#include "display/MultiPointCloud.hpp"

#include "io/Model.hpp"
#include "io/GridIO.hpp"

#include "../widgets/PointCloudTreeWidgetItem.h"
#include "../widgets/TriangleMeshTreeWidgetItem.h"
#include "../widgets/MultiPointCloudTreeWidgetItem.h"

#include "io/ModelFactory.hpp"
#include "io/DataStruct.hpp"

#include <boost/filesystem.hpp>
#include <boost/version.hpp>

using lvr::Model;
using lvr::GridIO;
using lvr::Grid;

DataCollectorFactory::DataCollectorFactory() {}

void DataCollectorFactory::create(string filename)
{
	// Get file extension
	boost::filesystem::path selectedFile(filename);

	string extension = selectedFile.extension().c_str();
	string name = selectedFile.filename().c_str();

	// Create a factory rto parse given file and extract loaders
	lvr::ModelFactory io;
    lvr::ModelPtr model = io.readModel( filename );

	if(model)
	{
        lvr::MeshBufferPtr    mesh_buffer  = model->m_mesh;
		lvr::PointBufferPtr   point_buffer = model->m_pointCloud;

	    if(mesh_buffer)
	    {
	        lvr::StaticMesh* mesh = new lssr::StaticMesh( model );
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

	    if(point_buffer)
	    {
	        if(point_buffer->getNumPoints() > 0)
	        {
	            // Check for multi point object
	            if(point_buffer->getSubClouds().size() > 1)
	            {
	                name = filename;
	                MultiPointCloud* pc = new MultiPointCloud(lvr::ModelPtr(model), name);
	                MultiPointCloudTreeWidgetItem* item = new MultiPointCloudTreeWidgetItem(MultiPointCloudItem);

	                // Setup supported render modes
	                int modes = 0;
	                size_t n_pn;
	                modes |= Points;
	                if(point_buffer->getPointNormalArray(n_pn))
	                {
	                    modes |= PointNormals;
	                }

	                item->setSupportedRenderModes(modes);
	                item->setViewCentering(false);
	                item->setName(name);
	                item->setRenderable(pc);

	                Static3DDataCollector* dataCollector = new Static3DDataCollector(pc, name, item);
	                Q_EMIT dataCollectorCreated( dataCollector );
	            }
	            else
	            {

	                PointCloud* pc = new PointCloud( model );
	                PointCloudTreeWidgetItem* item = new PointCloudTreeWidgetItem(PointCloudItem);

	                // Setup supported render modes
	                int modes = 0;
	                size_t n_pn;
	                modes |= Points;
	                if(point_buffer->getPointNormalArray(n_pn))
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
	}
	else
	{
	    // Try to load other objects
	    if(extension == ".grid")
	    {
	        GridIO io;
	        io.read( filename );
            size_t n_points, n_boxes;
            lvr::floatArr points = io.getPoints( n_points );
            lvr::uintArr  boxes  = io.getBoxes(  n_boxes );
	        if( points && boxes )
	        {
                Grid* grid = new Grid( points, boxes, n_points, n_boxes );
	            int modes = 0;
	            PointCloudTreeWidgetItem* item = new PointCloudTreeWidgetItem(PointCloudItem);
	            item->setSupportedRenderModes(modes);
	            item->setViewCentering(false);
	            item->setName("Grid");
	            item->setRenderable(grid);

	            Static3DDataCollector* dataCollector = new Static3DDataCollector(grid, "Grid", item);
	            Q_EMIT dataCollectorCreated( dataCollector );

	        }
	    }
	}

}



