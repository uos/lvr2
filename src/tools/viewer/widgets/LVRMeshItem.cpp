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
 * LVRMeshItem.cpp
 *
 *  @date Feb 11, 2014
 *  @author Thomas Wiemann
 */
#include "LVRMeshItem.hpp"
#include "LVRItemTypes.hpp"

namespace lvr
{

LVRMeshItem::LVRMeshItem(MeshBufferBridgePtr& ptr, QTreeWidgetItem* parent) :
        QTreeWidgetItem(parent, LVRMeshItemType), m_meshBridge(ptr)
{
    QIcon icon;
    icon.addFile(QString::fromUtf8(":/qv_mesh_tree_icon.png"), QSize(), QIcon::Normal, QIcon::Off);
    setIcon(0, icon);
    setText(0, "Triangle Mesh");
    setExpanded(true);

    // Setup Infos
    QTreeWidgetItem* vertItem = new QTreeWidgetItem(this);
    QString numVerts;
    vertItem->setText(0, "Num Points:");
    vertItem->setText(1, numVerts.setNum(ptr->getNumVertices()));
    addChild(vertItem);

    QTreeWidgetItem* faceItem = new QTreeWidgetItem(this);
    QString numFaces;
    faceItem->setText(0, "Num Triangles:");
    faceItem->setText(1, numFaces.setNum(ptr->getNumTriangles()));
    addChild(faceItem);
}

void LVRMeshItem::setColor(QColor &c)
{
    m_meshBridge->setBaseColor(c.redF(), c.greenF(), c.blueF());
}

LVRMeshItem::~LVRMeshItem()
{
    // TODO Auto-generated destructor stub
}

} /* namespace lvr */
