/**
 * MultiPointCloudDataCollector.h
 *
 *  @date 05.07.2011
 *  @author Thomas Wiemann
 */

#ifndef MULTIPOINTCLOUDDATACOLLECTOR_H_
#define MULTIPOINTCLOUDDATACOLLECTOR_H_

#include "Static3DDataCollector.h"
#include "../../lib/model3d/MultiPointCloud.h"

class MultiPointCloudDataCollector : public Static3DDataCollector
{
public:
    MultiPointCloudDataCollector(MultiPointCloud* renderable, string name, DataManager* manager, CustomTreeWidgetItem* item = 0);
    virtual ~MultiPointCloudDataCollector();
    virtual ViewerType supportedViewerType();
};

#endif /* MULTIPOINTCLOUDDATACOLLECTOR_H_ */
