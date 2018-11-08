/**
 * Copyright (c) 2018, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
#include "LVRTextureMeshItem.hpp"

#include <vtkSmartPointer.h>
#include <vtkActor.h>
#include <vtkPolyDataMapper.h>

namespace lvr2
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
    	if(bridge->m_meshBridge->hasTextures())
    	{
    		LVRTextureMeshItem* texItem = new LVRTextureMeshItem(bridge->m_meshBridge, this);
    		addChild(texItem);
    		texItem->setExpanded(true);
    	}
    	else
    	{
    		LVRMeshItem* meshItem = new LVRMeshItem(bridge->m_meshBridge, this);
    		addChild(meshItem);
    		meshItem->setExpanded(true);
    	}
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

void LVRModelItem::setPose(const Pose& pose)
{
    // Update vtk representation
    m_modelBridge->setPose(pose);

    // Update pose item
    m_poseItem->setPose(pose);
}

QString LVRModelItem::getName()
{
    return m_name;
}

void LVRModelItem::setName(QString name)
{
    m_name = name;
    setText(0, m_name);
}

ModelBridgePtr LVRModelItem::getModelBridge()
{
	return m_modelBridge;
}

bool LVRModelItem::isEnabled()
{
    return this->checkState(0);
}

void LVRModelItem::setVisibility(bool visible)
{
	m_modelBridge->setVisibility(visible);
}

void LVRModelItem::setModelVisibility(int column, bool globalValue)
{
	if(checkState(column) == globalValue || globalValue == true)
	{
	    setVisibility(checkState(column));
	}
}

LVRModelItem::~LVRModelItem()
{
    // TODO Auto-generated destructor stub
}

} /* namespace lvr2 */
