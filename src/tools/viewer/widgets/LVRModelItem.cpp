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
 * LVRModelItem.cpp
 *
 *  @date Feb 6, 2014
 *  @author Thomas Wiemann
 */
#include "LVRModelItem.hpp"
#include "LVRPointCloudItem.hpp"
#include "LVRMeshItem.hpp"
#include "LVRItemTypes.hpp"

#include <vtkSmartPointer.h>
#include <vtkActor.h>
#include <vtkPolyDataMapper.h>

namespace lvr
{

LVRModelItem::LVRModelItem(ModelBridgePtr bridge, QString name) :
    QTreeWidgetItem(LVRModelItemType), m_modelBridge(bridge), m_name(name)
{
    // Setup tree widget icon
    QIcon icon;
    icon.addFile(QString::fromUtf8(":/qv_model_tree_icon.png"), QSize(), QIcon::Normal, QIcon::Off);
    setIcon(0, icon);

    // Setup item properties
    setText(0, m_name);
    setCheckState(0, Qt::Checked);


    // Insert sub items
    if(bridge->m_pointBridge->getNumPoints())
    {
        LVRPointCloudItem* pointItem = new LVRPointCloudItem(bridge->m_pointBridge, this);
        addChild(pointItem);
        pointItem->setExpanded(true);
    }

    if(bridge->m_meshBridge->getNumTriangles())
    {
        LVRMeshItem* meshItem = new LVRMeshItem(bridge->m_meshBridge, this);
        addChild(meshItem);
        meshItem->setExpanded(true);
    }


    // Setup Pose
    m_poseItem = new LVRPoseItem(bridge, this);
    addChild(m_poseItem);
}

LVRModelItem::LVRModelItem(const LVRModelItem& item)
{
    m_modelBridge   = item.m_modelBridge;
    m_name          = item.m_name;
    m_poseItem      = item.m_poseItem;
}

Pose LVRModelItem::getPose()
{
    return m_poseItem->getPose();
}

void LVRModelItem::setPose( Pose& pose)
{
    // Update vtk representation
    m_modelBridge->setPose(pose);

    // Update pose item
    m_poseItem->setPose(pose);
}

ModelBridgePtr LVRModelItem::getModelBridge()
{
	return m_modelBridge;
}

LVRModelItem::~LVRModelItem()
{
    // TODO Auto-generated destructor stub
}

} /* namespace lvr */
