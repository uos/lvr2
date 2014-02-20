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
 * LVRPickItem.cpp
 *
 *  @date Feb 20, 2014
 *  @author Thomas Wiemann
 */
#include "LVRPickItem.hpp"

namespace lvr
{

LVRPickItem::LVRPickItem(QTreeWidget* parent, int type) :
        QTreeWidgetItem(parent, type)
{
    m_start = 0;
    m_end = 0;
    setText(0, "Empty");
    setText(1, "Empty");
}

LVRPickItem::~LVRPickItem()
{
   if(m_start) delete[] m_start;
   if(m_end) delete[] m_end;
}

void LVRPickItem::setStart(double* start)
{
    m_start = start;
    QString x, y, z, text;
    x.setNum(start[0], 'f');
    y.setNum(start[1], 'f');
    z.setNum(start[2], 'f');
    text = QString("%1 %2 %3").arg(x).arg(y).arg(z);
    setText(0, text);
}

void LVRPickItem::setEnd(double* end)
{
    m_end = end;
    QString x, y, z, text;
    x.setNum(end[0], 'f');
    y.setNum(end[1], 'f');
    z.setNum(end[2], 'f');
    text = QString("%1 %2 %3").arg(x).arg(y).arg(z);
    setText(1, text);
}

} /* namespace lvr */
