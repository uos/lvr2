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
 * LVRTreeWidgetHelper.cpp
 *
 *  @date Apr 10, 2014
 *  @author Thomas Wiemann
 */
#include "LVRTreeWidgetHelper.hpp"

#include "../widgets/LVRPointCloudItem.hpp"
#include "../widgets/LVRMeshItem.hpp"

#include <QTreeWidgetItemIterator>

namespace lvr2
{

LVRTreeWidgetHelper::LVRTreeWidgetHelper(QTreeWidget* tree)
{
   m_treeWidget = tree;
}

PointBufferPtr LVRTreeWidgetHelper::getPointBuffer(QString name)
{
    PointBufferPtr ptr;
    QTreeWidgetItemIterator m_it(m_treeWidget);
    while(*m_it)
    {
        if ( (*m_it)->type() == LVRModelItemType)
        {
            if( (*m_it)->text(0) == name)
            {
                cout << "Name check: " << (*m_it)->text(0).toStdString() << " " << name.toStdString() << endl;
                // Find point cloud sub item
                LVRModelItem* item = static_cast<LVRModelItem*>(*m_it);
                for(size_t i = 0; i < item->childCount(); i++)
                {
                    QTreeWidgetItem* treeItem = item->child(i);
                    if(treeItem && treeItem->type() == LVRPointCloudItemType)
                    {
                        LVRPointCloudItem* pointItem = static_cast<LVRPointCloudItem*>(treeItem);
                        ptr = pointItem->getPointBuffer();
                    }
                }
                break;
            }
        }
        ++m_it;
    }
    return ptr;
}

MeshBufferPtr LVRTreeWidgetHelper::getMeshBuffer(QString name)
{
    MeshBufferPtr ptr;
    QTreeWidgetItemIterator m_it(m_treeWidget);
    while(*m_it)
    {
        if ( (*m_it)->type() == LVRModelItemType)
        {
            if( (*m_it)->text(0) == name)
            {
                // Find point cloud sub item
                QTreeWidgetItemIterator sub_it(*m_it);
                while(*sub_it)
                {
                    if( (*sub_it)->type() == LVRMeshItemType)
                    {
                        LVRMeshItem* mesh_item = static_cast<LVRMeshItem*>(*sub_it);
                        ptr = mesh_item->getMeshBuffer();
                    }
                    ++sub_it;
                }
                break;
            }
            ++m_it;
        }
    }
    return ptr;
}

LVRModelItem* LVRTreeWidgetHelper::getModelItem(QString name)
{
    cout << "GET MODEL ITEM" << endl;
    LVRModelItem* ptr;
    QTreeWidgetItemIterator it(m_treeWidget);
    while (*it)
    {
        if ( (*it)->type() == LVRModelItemType)
        {
            if( (*it)->text(0) == name)
            {
                return static_cast<LVRModelItem*>(*it);
            }
        }
        ++it;
    }
    return ptr;
}

} /* namespace lvr2 */
