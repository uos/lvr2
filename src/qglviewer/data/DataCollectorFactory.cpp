/*
 * DataCollectorFactory.cpp
 *
 *  Created on: 07.10.2010
 *      Author: Thomas Wiemann
 */

#include "DataCollectorFactory.h"
#include "Static3DDataCollector.h"

#include "model3d/StaticMesh.h"
#include "model3d/PointCloud.h"

#include "../widgets/PointCloudTreeWidgetItem.h"

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
		dataCollector = new Static3DDataCollector(mesh, name, manager);
	}
	else if(extension == ".pts" || extension == ".xyz" || ".3d")
	{
		PointCloud* cloud = new PointCloud(filename);
		PointCloudTreeWidgetItem* item = new PointCloudTreeWidgetItem(PointCloudItem);
		item->setName(name);
		item->setNumPoints(cloud->points.size());

		dataCollector = new Static3DDataCollector(cloud, name, manager, item);

	}

	return dataCollector;
}



