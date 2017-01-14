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
 * LVRTreeWidgetHelper.cpp
 *
 *  @date Apr 10, 2014
 *  @author Thomas Wiemann
 */
#include "LVRTreeWidgetHelper.hpp"

#include "../widgets/LVRPointCloudItem.hpp"
#include "../widgets/LVRMeshItem.hpp"

#include <QTreeWidgetItemIterator>

namespace lvr
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

} /* namespace lvr */
