/*
 * DataCollectorFactory.cpp
 *
 *  Created on: 07.10.2010
 *      Author: Thomas Wiemann
 */

#include "DataCollectorFactory.h"
#include "Static3DDataCollector.h"
#include "MultiPointCloudDataCollector.h"

#include "model3d/StaticMesh.h"
#include "model3d/PointCloud.h"
#include "model3d/MultiPointCloud.h"

#include "../widgets/PointCloudTreeWidgetItem.h"
#include "../widgets/TriangleMeshTreeWidgetItem.h"

#include <boost/filesystem.hpp>
#include <boost/version.hpp>

DataCollectorFactory* DataCollectorFactory::m_instance = 0;

DataCollectorFactory* DataCollectorFactory::instance()
{
	if(DataCollectorFactory::m_instance == 0)
	{
		return new DataCollectorFactory;
	}
	else
	{
		return DataCollectorFactory::m_instance;
	}
}

DataCollector* DataCollectorFactory::create(string filename, DataManager* manager)
{
	// Get file extension
	boost::filesystem::path selectedFile(filename);
#if BOOST_VERSION < 104600
	string extension = selectedFile.extension();
	string name = selectedFile.filename();
#else
	string extension = selectedFile.extension().string();
	string name = selectedFile.filename();
#endif

	Static3DDataCollector* dataCollector = 0;

	// Try to load given file
	if(extension == ".ply")
	{
		StaticMesh* mesh = new StaticMesh(name);

		TriangleMeshTreeWidgetItem* item = new TriangleMeshTreeWidgetItem(TriangleMeshItem);
		item->setName(name);
		item->setNumFaces(mesh->getNumberOfFaces());
		item->setNumVertices(mesh->getNumberOfVertices());
		item->setRenderable(mesh);

		dataCollector = new Static3DDataCollector(mesh, name, manager, item);
	}
	else if(extension == ".pts" || extension == ".xyz" || extension == ".3d")
	{
	    // Create a point cloud object
		PointCloud* cloud = new PointCloud(filename);

		// Create and setup a tree widget item for the point cloud
		PointCloudTreeWidgetItem* item = new PointCloudTreeWidgetItem(PointCloudItem);
		item->setName(name);
		item->setNumPoints(cloud->points.size());
		item->setRenderable(cloud);

		// Create a new data collector
		dataCollector = new Static3DDataCollector(cloud, name, manager, item);
	}
	else
	{
	    MultiPointCloud* mpc = new MultiPointCloud(filename);
	    PointCloudTreeWidgetItem* item = new PointCloudTreeWidgetItem(PointCloudItem);
	    item->setName(name);
	    item->setNumPoints(0);
	    item->setRenderable(mpc);

	    dataCollector = new MultiPointCloudDataCollector(mpc, name, manager, item);

	}

	return dataCollector;
}



