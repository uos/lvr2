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

#include "VisualizerFactory.hpp"
#include "TriangleMeshVisualizer.hpp"
#include "PointCloudVisualizer.hpp"
#include "MultiPointCloudVisualizer.hpp"
#include "GridVisualizer.hpp"
#include "ClusterVisualizer.hpp"

#include "display/Grid.hpp"
#include "display/StaticMesh.hpp"
#include "display/PointCloud.hpp"
#include "display/MultiPointCloud.hpp"

#include "io/Model.hpp"
#include "io/GridIO.hpp"
#include "io/ModelFactory.hpp"
#include "io/DataStruct.hpp"

#include "../widgets/PointCloudTreeWidgetItem.h"
#include "../widgets/MultiPointCloudTreeWidgetItem.h"

#include <boost/filesystem.hpp>
#include <boost/version.hpp>

using lvr::Model;
using lvr::GridIO;
using lvr::Grid;

VisualizerFactory::VisualizerFactory() {}

void VisualizerFactory::create(string filename)
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
	    	TriangleMeshVisualizer* tmv = new TriangleMeshVisualizer(mesh_buffer, filename + " <mesh>");
	    	Q_EMIT visualizerCreated( tmv );
	    }

	    if(point_buffer)
	    {
	        if(point_buffer->getNumPoints() > 0)
	        {
	            // Check for multi point object
	            if(point_buffer->getSubClouds().size() > 1)
	            {
	                name = filename;


	                Visualizer* v = new MultiPointCloudVisualizer(point_buffer, name);
	                Q_EMIT visualizerCreated( v );
	            }
	            else
	            {
	                Visualizer* v = new PointCloudVisualizer(point_buffer, name);
	                Q_EMIT visualizerCreated( v );
	            }

	        }
	    }
	}
	else
	{
	    // Try to load other objects
	    if(extension == ".grid")
	    {
	           	Visualizer* v = new GridVisualizer(filename);
	           	if(v->renderable())
	           	{
	           		Q_EMIT visualizerCreated( v );
	           	}
	           	else
	           	{
	           		delete v;
	           	}
	    }

	    if(extension == ".clu")
	    {
	    	Visualizer* v = new ClusterVisualizer(filename);
	    	if(v->renderable())
	    	{
	    		Q_EMIT visualizerCreated( v );
	    	}
	    	else
	    	{
	    		delete v;
	    	}
	    }
	}

}



