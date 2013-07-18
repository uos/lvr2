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
 * MultiPointCloudVisualizer.cpp
 *
 *  Created on: 28.03.2012
 *      Author: Thomas Wiemann
 */

#include "MultiPointCloudVisualizer.hpp"

#include "display/MultiPointCloud.hpp"
#include "../widgets/MultiPointCloudTreeWidgetItem.h"

using namespace lvr;

MultiPointCloudVisualizer::MultiPointCloudVisualizer(PointBufferPtr buffer, string name = "<unnamed poind cloud>")
{
	MultiPointCloud* pc = new MultiPointCloud(buffer, name);
	m_renderable = pc;

	MultiPointCloudTreeWidgetItem* item = new MultiPointCloudTreeWidgetItem(MultiPointCloudItem);
	m_treeItem = item;

	// Setup supported render modes
	int modes = 0;
	size_t n_pn;
	modes |= Points;
	if(buffer->getPointNormalArray(n_pn))
	{
		modes |= PointNormals;
	}

	item->setSupportedRenderModes(modes);
	item->setViewCentering(false);
	item->setName(name);
	item->setRenderable(pc);
}

