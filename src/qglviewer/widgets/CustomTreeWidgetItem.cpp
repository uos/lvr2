/*
 * CustomTreeWidgetItem.cpp
 *
 *  Created on: 13.10.2010
 *      Author: Thomas Wiemann
 */

#include "CustomTreeWidgetItem.h"

CustomTreeWidgetItem::CustomTreeWidgetItem(int type)
: QTreeWidgetItem(type), m_oldCheckState(Qt::PartiallyChecked) , m_renderable(0){}

CustomTreeWidgetItem::CustomTreeWidgetItem(QTreeWidgetItem* parent, int type)
: QTreeWidgetItem(parent, type), m_oldCheckState(Qt::PartiallyChecked),  m_renderable(0) {}

CustomTreeWidgetItem::~CustomTreeWidgetItem() {}

bool CustomTreeWidgetItem::toggled()
{
	if(checkState(0) != m_oldCheckState)
	{
		m_oldCheckState = checkState(0);
		return true;
	}
	return false;
}

void CustomTreeWidgetItem::setInitialState(Qt::CheckState state)
{
	m_oldCheckState = state;
	setCheckState(0, state);
}
