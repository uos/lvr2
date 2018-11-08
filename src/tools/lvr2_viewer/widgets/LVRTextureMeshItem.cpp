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

/*
 * LVRTextureMeshItem.cpp
 *
 *  Created on: Dec 10, 2014
 *      Author: twiemann
 */

#include "LVRTextureMeshItem.hpp"

namespace lvr2
{

LVRTextureMeshItem::LVRTextureMeshItem(MeshBufferBridgePtr& ptr, QTreeWidgetItem* parent)
	: LVRMeshItem(ptr, parent)
{
	// Super class ctor should insert mesh information, add texture
	// information here
	addSubItems();
}

void LVRTextureMeshItem::addSubItems()
{
	  QIcon icon;
	  icon.addFile(QString::fromUtf8(":/qv_mesh_texture_icon.png"), QSize(), QIcon::Normal, QIcon::Off);

	  // Add item to parent to display them on the same tree level
	  QTreeWidgetItem* item = new QTreeWidgetItem(m_parent);
	  item->setIcon(0, icon);
	  item->setText(0, "Textures");


	  // Add sub items containing statistics
	  QTreeWidgetItem* texNumItem = new QTreeWidgetItem(item);
	  QString numTextures;
	  texNumItem->setText(0, "Num Textures:");
	  texNumItem->setText(1, numTextures.setNum(m_meshBridge->getNumTextures()));
	  item->addChild(texNumItem);

	  QTreeWidgetItem* colFaceItem = new QTreeWidgetItem(item);
	  QString numColFaces;
	  colFaceItem->setText(0, "Num Colored Tris:");
	  colFaceItem->setText(1, numColFaces.setNum(m_meshBridge->getNumColoredFaces()));
	  item->addChild(colFaceItem);

	  QTreeWidgetItem* texFaceItem = new QTreeWidgetItem(item);
	  QString numTexFaces;
	  texFaceItem->setText(0, "Num Textured Tris:");
	  texFaceItem->setText(1, numTexFaces.setNum(m_meshBridge->getNumTexturedFaces()));
	  item->addChild(texFaceItem);

	  m_parent->addChild(item);
}

LVRTextureMeshItem::~LVRTextureMeshItem()
{
	// TODO Auto-generated destructor stub
}

} /* namespace lvr2 */
