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
 * KinectPointCloudVisualizer.cpp
 *
 *  Created on: 28.03.2012
 *      Author: Thomas Wiemann
 */

#include "KinectPointCloudVisualizer.hpp"

#include "../widgets/PointCloudTreeWidgetItem.h"


KinectPointCloudVisualizer::KinectPointCloudVisualizer()
{
	PointCloudTreeWidgetItem* item = new PointCloudTreeWidgetItem(PointCloudItem);

	// Setup supported render modes
	int modes = 0;
	modes |= Points;

	m_pointCloud = new InteractivePointCloud;
	m_renderable = m_pointCloud;

	item->setSupportedRenderModes(modes);
	item->setViewCentering(false);
	item->setName("Kinect Data");
	item->setNumPoints(640 * 480);
	item->setRenderable(m_pointCloud);

	m_treeItem = item;

	//start();
}

void KinectPointCloudVisualizer::run()
{
	while(true)
	{
		usleep(1000);
	}
}

void KinectPointCloudVisualizer::updateBuffer(PointBufferPtr* buffer)
{
	m_pointCloud->updateBuffer(*buffer);
}
