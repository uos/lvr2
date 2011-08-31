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

//	// Try to load given file
//	if(extension == ".ply")
//	{
//		StaticMesh* mesh = new StaticMesh(name);
//
//		TriangleMeshTreeWidgetItem* item = new TriangleMeshTreeWidgetItem(TriangleMeshItem);
//		item->setName(name);
//		item->setViewCentering(true);
//		item->setNumFaces(mesh->getNumberOfFaces());
//		item->setNumVertices(mesh->getNumberOfVertices());
//		item->setRenderable(mesh);
//
//		dataCollector = new Static3DDataCollector(mesh, name, item);
//	}
//	else if(extension == ".pts" || extension == ".xyz" || extension == ".3d")
//	{
//	    // Create a point cloud object
//		PointCloud* cloud = new PointCloud(filename);
//
//		// Create and setup a tree widget item for the point cloud
//		PointCloudTreeWidgetItem* item = new PointCloudTreeWidgetItem(PointCloudItem);
//		item->setViewCentering(true);
//		item->setName(name);
//		item->setNumPoints(cloud->points.size());
//		item->setRenderable(cloud);
//
//		// Create a new data collector
//		dataCollector = new Static3DDataCollector(cloud, name, item);
//	}
//	else
//	{
//	    MultiPointCloud* mpc = new MultiPointCloud(filename);
//	    MultiPointCloudTreeWidgetItem* item = new MultiPointCloudTreeWidgetItem(MultiPointCloudItem);
//
//	    // Set label etc.
//	    item->setViewCentering(true);
//	    item->setName(filename);
//	    item->setRenderable(mpc);
//	    dataCollector = new Static3DDataCollector(mpc, name, item);
//
//	}

	// Create a factory rto parse given file and extract loaders
	lssr::IOFactory io(filename);
	MeshLoader*   mesh_loader  = io.getMeshLoader();
	PointLoader*  point_loader = io.getPointLoader();

	if(mesh_loader)
	{
	    lssr::StaticMesh* mesh = new lssr::StaticMesh(*mesh_loader);
	   //TriangleTreeWidgetItem item =
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
	        item->setViewCentering(true);
	        item->setName(name);
	        item->setNumPoints(pc->m_points.size());
	        item->setRenderable(pc);

	        Static3DDataCollector* dataCollector = new Static3DDataCollector(pc, name, item);

	        cout << "EMIT" << endl;
	        Q_EMIT dataCollectorCreated( dataCollector );

	    }
	}

}



