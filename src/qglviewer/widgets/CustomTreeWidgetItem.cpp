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


 /*
 * CustomTreeWidgetItem.cpp
 *
 *  Created on: 13.10.2010
 *      Author: Thomas Wiemann
 */

#include "CustomTreeWidgetItem.h"

CustomTreeWidgetItem::CustomTreeWidgetItem(int type)
: QTreeWidgetItem(type), m_oldCheckState(Qt::PartiallyChecked) , m_renderable(0), m_centerOnClick(false){}

CustomTreeWidgetItem::CustomTreeWidgetItem(QTreeWidgetItem* parent, int type)
: QTreeWidgetItem(parent, type), m_oldCheckState(Qt::PartiallyChecked),  m_renderable(0), m_centerOnClick(false) {}

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

void CustomTreeWidgetItem::setName(string name)
{
    m_name = name;
    setText(0, QString(m_name.c_str()));
}

bool CustomTreeWidgetItem::supportsMode(int mode)
{
    return (m_renderMode & mode);
}
