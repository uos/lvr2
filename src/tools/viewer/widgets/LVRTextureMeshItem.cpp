/*
 * LVRTextureMeshItem.cpp
 *
 *  Created on: Dec 10, 2014
 *      Author: twiemann
 */

#include "LVRTextureMeshItem.hpp"

namespace lvr
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

} /* namespace lvr */
