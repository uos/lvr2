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
 * LVRPoseItem.cpp
 *
 *  @date Feb 17, 2014
 *  @author Thomas Wiemann
 */
#include "LVRPoseItem.hpp"
#include "LVRItemTypes.hpp"

namespace lvr
{

LVRPoseItem::LVRPoseItem(ModelBridgePtr bridge, QTreeWidgetItem* parent):
        QTreeWidgetItem(parent, LVRPoseItemType)
{
    m_pose = bridge->getPose();

    // Setup
    QIcon icon;
    icon.addFile(QString::fromUtf8(":/qv_transform_tree_icon.png"), QSize(), QIcon::Normal, QIcon::Off);
    setIcon(0, icon);
    setText(0, "Pose");

    m_xItem = new QTreeWidgetItem(this);
    m_yItem = new QTreeWidgetItem(this);
    m_zItem = new QTreeWidgetItem(this);
    m_rItem = new QTreeWidgetItem(this);
    m_tItem = new QTreeWidgetItem(this);
    m_pItem = new QTreeWidgetItem(this);

    addChild(m_xItem);
    addChild(m_yItem);
    addChild(m_zItem);
    addChild(m_rItem);
    addChild(m_tItem);
    addChild(m_pItem);

    setPose(m_pose);
}


void LVRPoseItem::setPose(const Pose& pose)
{
    m_pose = pose;
    QString num;

    m_xItem->setText(0, "Position X:");
    m_xItem->setText(1, num.setNum(m_pose.x,'F'));

    m_yItem->setText(0, "Position Y:");
    m_yItem->setText(1, num.setNum(m_pose.y,'f'));

    m_zItem->setText(0, "Position Z:");
    m_zItem->setText(1, num.setNum(m_pose.z,'f'));

    m_rItem->setText(0, "Rotation X:");
    m_rItem->setText(1, num.setNum(m_pose.r,'f'));

    m_tItem->setText(0, "Rotation Y:");
    m_tItem->setText(1, num.setNum(m_pose.t,'f'));

    m_pItem->setText(0, "Rotation Z:");
    m_pItem->setText(1, num.setNum(m_pose.p,'f'));

}

Pose LVRPoseItem::getPose()
{
    return m_pose;
}

LVRPoseItem::~LVRPoseItem()
{
    // TODO Auto-generated destructor stub
}

} /* namespace lvr */
