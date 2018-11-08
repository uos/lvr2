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
 * LVRMeshItem.cpp
 *
 *  @date Feb 11, 2014
 *  @author Thomas Wiemann
 */
#include "LVRMeshItem.hpp"
#include "LVRItemTypes.hpp"

namespace lvr2
{

LVRMeshItem::LVRMeshItem(MeshBufferBridgePtr& ptr, QTreeWidgetItem* parent) :
        QTreeWidgetItem(parent, LVRMeshItemType), m_meshBridge(ptr)
{
    // set initial values
    m_opacity = 1;
    m_color = QColor(255,255,255);
    m_visible = true;
    m_shader = 0;
    m_parent = parent;
    addSubItems();
}

void LVRMeshItem::addSubItems()
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
    vertItem->setText(1, numVerts.setNum(m_meshBridge->getNumVertices()));
    addChild(vertItem);

    QTreeWidgetItem* faceItem = new QTreeWidgetItem(this);
    QString numFaces;
    faceItem->setText(0, "Num Triangles:");
    faceItem->setText(1, numFaces.setNum(m_meshBridge->getNumTriangles()));
    addChild(faceItem);
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

void LVRMeshItem::setSelectionColor(QColor &c)
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

vtkSmartPointer<vtkActor> LVRMeshItem::getWireframeActor()
{
    return m_meshBridge->getWireframeActor();
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

} /* namespace lvr2 */
