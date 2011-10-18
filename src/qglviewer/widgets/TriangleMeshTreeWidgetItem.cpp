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
 * TriangleMeshTreeWidgetItem.cpp
 *
 *  @date 16.06.2011
 *  @author Thomas Wiemann
 */

#include "TriangleMeshTreeWidgetItem.h"

#include <sstream>
using std::stringstream;

TriangleMeshTreeWidgetItem::TriangleMeshTreeWidgetItem(int type)
     : CustomTreeWidgetItem(type)
{
   m_numVertices = 0;
   m_numFaces = 0;
   setInitialState(Qt::Checked);
}

TriangleMeshTreeWidgetItem::TriangleMeshTreeWidgetItem(QTreeWidgetItem* parent, int type)
    : CustomTreeWidgetItem(parent, type)
{
    m_numVertices = 0;
    m_numFaces = 0;
    setInitialState(Qt::Checked);
}


void TriangleMeshTreeWidgetItem::setNumVertices(size_t n)
{
    m_numVertices = n;

    // Create new item
    QTreeWidgetItem* vertItem = new QTreeWidgetItem(this);

    // Create label text
    stringstream pstream;
    pstream << "Vertices: " << m_numVertices;

    // Set text and add child
    vertItem->setText(0, pstream.str().c_str());
    addChild(vertItem);
}

void TriangleMeshTreeWidgetItem::setNumFaces(size_t n)
{
    m_numFaces = n;

    // Create new item
    QTreeWidgetItem* faceItem = new QTreeWidgetItem(this);

    // Create label text
    stringstream pstream;
    pstream << "Faces: " << m_numVertices;

    // Set text and add child
    faceItem->setText(0, pstream.str().c_str());
    addChild(faceItem);
}
