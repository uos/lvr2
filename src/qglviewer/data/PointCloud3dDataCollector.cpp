/*
 * PointCloud3dDataCollector.cpp
 *
 *  Created on: 15.10.2010
 *      Author: Thomas Wiemann
 */
#include <lib3d/PointCloud.h>

#include "PointCloud3dDataCollector.h"
#include "DataManager.h"

PointCloud3dDataCollector::PointCloud3dDataCollector(ClientProxy* proxy, DataManager* manager)
	: ThreadedDataCollector(proxy, manager)
{
	m_renderable = new PointCloud;
	start();
}

PointCloud3dDataCollector::~PointCloud3dDataCollector()
{
	// TODO Auto-generated destructor stub
}

void PointCloud3dDataCollector::run()
{
	Pointcloud3dProxy* proxy = static_cast<Pointcloud3dProxy*>(m_proxy);
	proxy->NotFresh();
	cout << "PPROXY: " <<  m_proxy << " " << proxy << endl;
	PlayerClient* client = proxy->GetPlayerClient();
	while(true)
	{
		client->ReadIfWaiting();

		if(proxy->IsFresh() && proxy->IsValid())		{
			proxy->NotFresh();
			updatePointCloudData(proxy);

		}
		usleep(10000);
	}
}

void PointCloud3dDataCollector::updatePointCloudData(Pointcloud3dProxy* proxy)
{
	QMutexLocker locker(&m_mutex);

	PointCloud* pointCloud = static_cast<PointCloud*>(m_renderable);
	pointCloud->clear();
	size_t count = proxy->GetCount();

	for(size_t i = 0; i < count; i++)
	{
		player_pointcloud3d_element_t pt = proxy->GetPoint(i);
		cout << pt.point.px << " " <<  pt.point.py << " " << pt.point.pz << endl;
		pointCloud->addPoint(
				(float)pt.point.px,
				(float)pt.point.pz,
				-(float)pt.point.py,
				pt.color.red,
				pt.color.green,
				pt.color.blue);
	}

	m_manager->update(this);
}

ViewerType  PointCloud3dDataCollector::supportedViewerType()
{
	return PERSPECTIVE_VIEWER;
}
