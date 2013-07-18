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
 * GridVisualizer.cpp
 *
 *  Created on: 28.03.2012
 *      Author: Thomas Wiemann
 */

#include "display/Grid.hpp"
#include "io/GridIO.hpp"
#include "GridVisualizer.hpp"

#include "../widgets/PointCloudTreeWidgetItem.h"

GridVisualizer::GridVisualizer(string filename)
{
	lvr::GridIO io;
	io.read( filename );
	size_t n_points, n_boxes;
	lvr::floatArr points = io.getPoints( n_points );
	lvr::uintArr  boxes  = io.getBoxes(  n_boxes );
	if( points && boxes )
	{
		lvr::Grid* grid = new lvr::Grid( points, boxes, n_points, n_boxes );
		m_renderable = grid;

		int modes = 0;
		PointCloudTreeWidgetItem* item = new PointCloudTreeWidgetItem(PointCloudItem);
		item->setSupportedRenderModes(modes);
		item->setViewCentering(false);
		item->setName("Grid");
		item->setRenderable(grid);
		m_treeItem = item;
	}
}

