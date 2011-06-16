/**
 * PointCloudTreeWidgetItem.h
 *
 *  @date 16.06.2011
 *  @author Thomas Wiemann
 */

#ifndef POINTCLOUDTREEWIDGETITEM_H_
#define POINTCLOUDTREEWIDGETITEM_H_

#include "CustomTreeWidgetItem.h"

#include <string>
using std::string;

class PointCloudTreeWidgetItem : public CustomTreeWidgetItem
{
public:
    PointCloudTreeWidgetItem(int type);
    PointCloudTreeWidgetItem(QTreeWidgetItem* parent, int type);

    virtual ~PointCloudTreeWidgetItem() {};

    void setNumPoints(size_t numPoints);

private:

    void addChildren();

    size_t      m_numPoints;
    string      m_name;
    bool        m_hasColor;
};

#endif /* POINTCLOUDTREEWIDGETITEM_H_ */
