/* Copyright (C) 2011 Uni Osnabrück
 * This file is part of the LAS VEGAS Reconstruction Toolkit,
 *
 * LAS VEGAS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * LAS VEGAS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA
 */

/**
 * LVRPointCloudItem.cpp
 *
 *  @date Feb 7, 2014
 *  @author Thomas Wiemann
 */
#include "LVRPointCloudItem.hpp"
#include "LVRItemTypes.hpp"

namespace lvr
{

LVRPointCloudItem::LVRPointCloudItem(PointBufferBridgePtr& ptr, QTreeWidgetItem* item) :
       QTreeWidgetItem(item, LVRPointCloudItemType), m_parent(item), m_pointBridge(ptr)
{
    // Setup tree widget icon
    QIcon icon;
    icon.addFile(QString::fromUtf8(":/qv_pc_tree_icon.png"), QSize(), QIcon::Normal, QIcon::Off);
    setIcon(0, icon);
    setText(0, "Point Cloud");
    setExpanded(true);

    // Setup Infos
    QTreeWidgetItem* numItem = new QTreeWidgetItem(this);
    QString num;
    numItem->setText(0, "Num Points:");
    numItem->setText(1, num.setNum(ptr->getNumPoints()));
    addChild(numItem);

    QTreeWidgetItem* normalItem = new QTreeWidgetItem(this);
    normalItem->setText(0, "Has Normals:");
    if(ptr->hasNormals())
    {
        normalItem->setText(1, "yes");
    }
    else
    {
        normalItem->setText(1, "no");
    }
    addChild(normalItem);

    QTreeWidgetItem* colorItem = new QTreeWidgetItem(this);
    colorItem->setText(0, "Has Colors:");
    if(ptr->hasColors())
    {
        colorItem->setText(1, "yes");
    }
    else
    {
        colorItem->setText(1, "no");
    }
    addChild(colorItem);
}

void LVRPointCloudItem::setColor(QColor &c)
{
    m_pointBridge->setBaseColor(c.redF(), c.greenF(), c.blueF());
}

LVRPointCloudItem::~LVRPointCloudItem()
{
    // TODO Auto-generated destructor stub
}

} /* namespace lvr */
