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
 * LVRPoseItem.cpp
 *
 *  @date Feb 17, 2014
 *  @author Thomas Wiemann
 */
#include "LVRPoseItem.hpp"
#include "LVRItemTypes.hpp"

namespace lvr2
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

} /* namespace lvr2 */
