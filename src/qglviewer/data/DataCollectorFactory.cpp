/*
 * DataCollectorFactory.cpp
 *
 *  Created on: 07.10.2010
 *      Author: Thomas Wiemann
 */

#include "DataCollectorFactory.h"
#include "Static3DDataCollector.h"
#include "PointCloud3dDataCollector.h"

#include <lib3d/StaticMesh.h>
#include <lib3d/PointCloud.h>

#include <boost/filesystem.hpp>

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
	string extension = selectedFile.extension();

	Static3DDataCollector* dataCollector = 0;

	// Try to load given file
	if(extension == ".ply")
	{
		StaticMesh* mesh = new StaticMesh(filename);
		dataCollector = new Static3DDataCollector(mesh, filename, manager);
	}
	else if(extension == ".pts" || extension == ".xyz" || ".3d")
	{
		PointCloud* cloud = new PointCloud(filename);
		dataCollector = new Static3DDataCollector(cloud, filename, manager);
	}

	return dataCollector;
}

DataCollector* DataCollectorFactory::create(ClientProxy* proxy, DataManager* manager)
{

	int interface = proxy->GetInterface();

	switch(interface)
	{
	case  61: return new PointCloud3dDataCollector(proxy, manager);
	case 128: return new Model3dDataCollector(proxy, manager);
	default:
		cout << "DataCollectorFactory::create(): warning: Unknown interface." << endl;
		return 0;
	}
}


