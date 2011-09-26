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
#include "io/PointLoader.hpp"

#include "../widgets/PointCloudTreeWidgetItem.h"
#include "../widgets/TriangleMeshTreeWidgetItem.h"
#include "../widgets/MultiPointCloudTreeWidgetItem.h"

#include "io/IOFactory.hpp"

#include <boost/filesystem.hpp>
#include <boost/version.hpp>

using lssr::MeshLoader;
using lssr::PointLoader;

DataCollectorFactory::DataCollectorFactory() {}

void DataCollectorFactory::create(string filename)
{
	// Get file extension
	boost::filesystem::path selectedFile(filename);

	string extension = selectedFile.extension().c_str();
	string name = selectedFile.filename().c_str();

	// Create a factory rto parse given file and extract loaders
	lssr::IOFactory io(filename);
	MeshLoader*   mesh_loader  = io.getMeshLoader();
	PointLoader*  point_loader = io.getPointLoader();

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
	        modes |= Points;
	        if(point_loader->getPointNormalArray())
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



