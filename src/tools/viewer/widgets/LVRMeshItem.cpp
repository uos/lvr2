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
    m_meshBridge->setShading(0);

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

    // set initial values
    m_opacity = 1;
    m_color = QColor(255,255,255);
    m_visible = true;
    m_shader = 0;
}

QColor LVRMeshItem::getColor()
{
	return m_color;
}

void LVRMeshItem::setColor(QColor &c)
{
    m_color = c;
    m_meshBridge->setBaseColor(c.redF(), c.greenF(), c.blueF());
}

void LVRMeshItem::setSelectionColor(QColor& c)
{
    m_meshBridge->setBaseColor(c.redF(), c.greenF(), c.blueF());
}

void LVRMeshItem::resetColor()
{
    m_meshBridge->setBaseColor(m_color.redF(), m_color.greenF(), m_color.blueF());
}

float LVRMeshItem::getOpacity()
{
	return m_opacity;
}

void LVRMeshItem::setOpacity(float &opacity)
{
    m_meshBridge->setOpacity(opacity);
    m_opacity = opacity;
}

bool LVRMeshItem::getVisibility()
{
	return m_visible;
}

void LVRMeshItem::setVisibility(bool &visibility)
{
	m_meshBridge->setVisibility(visibility);
	m_visible = visibility;
}

int LVRMeshItem::getShading()
{
    return m_shader;
}

void LVRMeshItem::setShading(int &shader)
{
    m_meshBridge->setShading(shader);
    m_shader = shader;
}

MeshBufferPtr LVRMeshItem::getMeshBuffer()
{
    return m_meshBridge->getMeshBuffer();
}

vtkSmartPointer<vtkActor> LVRMeshItem::getActor()
{
	return m_meshBridge->getMeshActor();
}

LVRMeshItem::~LVRMeshItem()
{
    // TODO Auto-generated destructor stub
}

} /* namespace lvr */
