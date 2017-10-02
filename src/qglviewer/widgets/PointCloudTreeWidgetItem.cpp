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
 * PointCloudTreeWidgetItem.cpp
 *
 *  @date 16.06.2011
 *  @author Thomas Wiemann
 */

#include "PointCloudTreeWidgetItem.h"
#include <sstream>
using std::stringstream;

PointCloudTreeWidgetItem::PointCloudTreeWidgetItem(int type) : CustomTreeWidgetItem(type)
{
    m_name = "undefined";
    m_numPoints = 0;
    setText(0, QString(m_name.c_str()));
    setInitialState(Qt::Checked);
}

PointCloudTreeWidgetItem::PointCloudTreeWidgetItem(QTreeWidgetItem* parent, int type)
    : CustomTreeWidgetItem(parent, type)
{
    m_name = "undefined";
    m_numPoints = 0;

    setText(0, QString(m_name.c_str()));
    setInitialState(Qt::Checked);
}

void PointCloudTreeWidgetItem::setNumPoints(size_t numPoints)
{
    m_numPoints = numPoints;

    // Create new item
    QTreeWidgetItem* pointsItem = new QTreeWidgetItem(this);

    // Create label text
    stringstream pstream;
    pstream << "Points: " << m_numPoints;

    // Set text and add child
    pointsItem->setText(0, pstream.str().c_str());
    addChild(pointsItem);
}
