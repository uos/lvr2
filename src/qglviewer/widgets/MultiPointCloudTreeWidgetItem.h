/**
 * MultiPointCloudTreeWidgetItem.h
 *
 *  @date 07.07.2011
 *  @author Thomas Wiemann
 */

#ifndef MULTIPOINTCLOUDTREEWIDGETITEM_H_
#define MULTIPOINTCLOUDTREEWIDGETITEM_H_

#include "model3d/PointCloud.h"
#include "model3d/MultiPointCloud.h"

#include "CustomTreeWidgetItem.h"

class MultiPointCloudTreeWidgetItem: public CustomTreeWidgetItem
{
public:
    MultiPointCloudTreeWidgetItem(int type);
    MultiPointCloudTreeWidgetItem(QTreeWidgetItem* parent, int type);

    void setRenderable(MultiPointCloud* pc);

    virtual ~MultiPointCloudTreeWidgetItem();
};

#endif /* MULTIPOINTCLOUDTREEWIDGETITEM_H_ */
