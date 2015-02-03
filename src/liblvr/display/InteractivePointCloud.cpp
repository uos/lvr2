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
 * InteractivePointCloud.cpp
 *
 *  Created on: 02.04.2012
 *      Author: Thomas Wiemann
 */

#include "display/InteractivePointCloud.hpp"

namespace lvr
{

InteractivePointCloud::InteractivePointCloud()
{
	m_boundingBox = new BoundingBox<Vertex<float> >(
			Vertex<float>(-8, -8, -8),
			Vertex<float>(8, 8, 8)
			);

	updateBuffer(PointBufferPtr());
}

InteractivePointCloud::InteractivePointCloud(PointBufferPtr buffer)
{
	m_boundingBox = new BoundingBox<Vertex<float> >(
			Vertex<float>(-8, -8, -8),
			Vertex<float>(8, 8, 8)
			);

	updateBuffer(buffer);
}


InteractivePointCloud::~InteractivePointCloud()
{

}

void InteractivePointCloud::render()
{
	if(m_buffer)
	{
		glColor3f(1.0, 0.0, 0.0);
		glDrawArrays(GL_POINTS, 0, m_buffer->getNumPoints());
	}
}

void InteractivePointCloud::updateBuffer(PointBufferPtr buffer)
{
	if(buffer)
	{
		if(!m_boundingBox)
		{
			m_boundingBox = new BoundingBox<Vertex<float> >;
			m_boundingBox->expand(8000, 8000, 8000);
		}

		size_t num_vertices;
		float* vertices = buffer->getPointArray(num_vertices).get();

//		m_boundingBox = new BoundingBox<Vertex<float> >;
//		for (int i = 0; i < int(num_vertices); i++)
//		{
//			int index = 3 * i;
//			m_boundingBox->expand(vertices[index], vertices[index + 1], vertices[index + 2]);
//		}

		glVertexPointer(3, GL_FLOAT, 0, vertices);
		m_buffer = buffer;
	}
}

} /* namespace lvr */
