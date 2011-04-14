/*
 * PointCloud3dDataCollector.h
 *
 *  Created on: 15.10.2010
 *      Author: Thomas Wiemann
 */

#ifndef POINTCLOUD3DDATACOLLECTOR_H_
#define POINTCLOUD3DDATACOLLECTOR_H_

#include "ThreadedDataCollector.h"
#include <libplayerc++/playerc++.h>

using namespace PlayerCc;

class PointCloud3dDataCollector : public ThreadedDataCollector
{
public:
	PointCloud3dDataCollector(ClientProxy* proxy, DataManager* manager);
	virtual ~PointCloud3dDataCollector();

	virtual ViewerType supportedViewerType();
	virtual void run();


private:
	void updatePointCloudData(Pointcloud3dProxy* proxy);

};

#endif /* POINTCLOUD3DDATACOLLECTOR_H_ */
