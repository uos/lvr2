/**
 * MultiPointCloudDataCollector.cpp
 *
 *  @date 05.07.2011
 *  @author Thomas Wiemann
 */

#include "MultiPointCloudDataCollector.h"

MultiPointCloudDataCollector::MultiPointCloudDataCollector(MultiPointCloud* renderable, string name, DataManager* manager, CustomTreeWidgetItem* item)
    : Static3DDataCollector(renderable, name, manager, item) {}

MultiPointCloudDataCollector::~MultiPointCloudDataCollector()
{
    // TODO Auto-generated destructor stub
}

ViewerType MultiPointCloudDataCollector::supportedViewerType()
{
    return PERSPECTIVE_VIEWER;
}
