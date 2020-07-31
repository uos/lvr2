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
 * LVRLabelItem.cpp
 *
 *  @date Feb 6, 2014
 *  @author Thomas Wiemann
 */
#include "LVRLabelItem.hpp"
#include "LVRPointCloudItem.hpp"
#include "LVRItemTypes.hpp"
#include "LVRTextureMeshItem.hpp"

#include <vtkSmartPointer.h>
#include <vtkActor.h>
#include <vtkPolyDataMapper.h>

namespace lvr2
{

LVRLabelItem::LVRLabelItem(LabelBridgePtr bridge, QString name) :
    QTreeWidgetItem(LVRLabelItemType), m_labelBridge(bridge), m_name(name)
{

    // Setup item properties
    setText(0, m_name);
    setCheckState(0, Qt::Checked);

    // Insert sub items
    if(bridge->getPointBridge() && bridge->getPointBridge()->getNumPoints())
    {
        LVRPointCloudItem* pointItem = new LVRPointCloudItem(bridge->getPointBridge(), this);
        addChild(pointItem);
        pointItem->setExpanded(true);
    }
}

LVRLabelItem::LVRLabelItem(const LVRLabelItem& item)
{
    m_labelBridge   = item.m_labelBridge;
    m_name          = item.m_name;
}

QString LVRLabelItem::getName()
{
    return m_name;
}

void LVRLabelItem::setName(QString name)
{
    m_name = name;
    setText(0, m_name);
}

LabelBridgePtr LVRLabelItem::getLabelBridge()
{
	return m_labelBridge;
}

bool LVRLabelItem::isEnabled()
{
    return this->checkState(0);
}

void LVRLabelItem::setVisibility(bool visible)
{
	m_labelBridge->setVisibility(visible);
}

LVRLabelItem::~LVRLabelItem()
{
    // TODO Auto-generated destructor stub
}

} /* namespace lvr2 */
