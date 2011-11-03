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
 * MultiPointCloudTreeWidgetItem.cpp
 *
 *  @date 07.07.2011
 *  @author Thomas Wiemann
 */

#include "MultiPointCloudTreeWidgetItem.h"
#include "PointCloudTreeWidgetItem.h"

MultiPointCloudTreeWidgetItem::MultiPointCloudTreeWidgetItem(int type)
     : CustomTreeWidgetItem(type)
{
    // TODO Auto-generated constructor stub

}

MultiPointCloudTreeWidgetItem::MultiPointCloudTreeWidgetItem(QTreeWidgetItem* parent, int type)
     : CustomTreeWidgetItem(parent, type)
{
    // TODO Auto-generated destructor stub
}

MultiPointCloudTreeWidgetItem::~MultiPointCloudTreeWidgetItem()
{

}


void MultiPointCloudTreeWidgetItem::setRenderable(MultiPointCloud* mpc)
{
    m_renderable = mpc;

    // Add stored point clouds as sub widgets
    lssr::pc_attr_it it;

    for(it = mpc->first(); it != mpc->last(); it ++)
    {
        PointCloud* pc = it->first;
        PointCloudTreeWidgetItem* item = new PointCloudTreeWidgetItem(PointCloudItem);

        // Setup supported render modes
        int modes = 0;
        size_t n_pn;
        modes |= Points;

        item->setName(pc->Name());
        item->setNumPoints(pc->m_points.size());
        item->setRenderable(pc);

        addChild(item);
    }
}
