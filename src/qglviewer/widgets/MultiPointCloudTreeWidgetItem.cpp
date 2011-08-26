/**
 * MultiPointCloudTreeWidgetItem.cpp
 *
 *  @date 07.07.2011
 *  @author Thomas Wiemann
 */

#include "MultiPointCloudTreeWidgetItem.h"
#include "PointCloudTreeWidgetItem.h"

MultiPointCloudTreeWidgetItem::MultiPointCloudTreeWidgetItem(int type)
     : CustomTreeWidgetItem(type)
{
    // TODO Auto-generated constructor stub

}

MultiPointCloudTreeWidgetItem::MultiPointCloudTreeWidgetItem(QTreeWidgetItem* parent, int type)
     : CustomTreeWidgetItem(parent, type)
{
    // TODO Auto-generated destructor stub
}

MultiPointCloudTreeWidgetItem::~MultiPointCloudTreeWidgetItem()
{

}


void MultiPointCloudTreeWidgetItem::setRenderable(MultiPointCloud* mpc)
{
    m_renderable = mpc;

    // Add stored point clouds as sub widgets
    lssr::pc_attr_it it;

    for(it = mpc->first(); it != mpc->last(); it ++)
    {
        PointCloud* pc = it->first;
        PointCloudTreeWidgetItem* item = new PointCloudTreeWidgetItem(PointCloudItem);
        item->setName(pc->Name());
        item->setNumPoints(pc->m_points.size());
        item->setRenderable(pc);

        addChild(item);
    }
}
