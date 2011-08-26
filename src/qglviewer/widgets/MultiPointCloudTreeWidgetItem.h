/**
 * MultiPointCloudTreeWidgetItem.h
 *
 *  @date 07.07.2011
 *  @author Thomas Wiemann
 */

#ifndef MULTIPOINTCLOUDTREEWIDGETITEM_H_
#define MULTIPOINTCLOUDTREEWIDGETITEM_H_

#include "display/PointCloud.hpp"
#include "display/MultiPointCloud.hpp"

#include "CustomTreeWidgetItem.h"

using lssr::MultiPointCloud;
using lssr::PointCloud;

class MultiPointCloudTreeWidgetItem: public CustomTreeWidgetItem
{
public:
    MultiPointCloudTreeWidgetItem(int type);
    MultiPointCloudTreeWidgetItem(QTreeWidgetItem* parent, int type);

    void setRenderable(MultiPointCloud* pc);

    virtual ~MultiPointCloudTreeWidgetItem();
};

#endif /* MULTIPOINTCLOUDTREEWIDGETITEM_H_ */
