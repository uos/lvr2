/**
 * Copyright (c) 2018, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * InteractivePointCloud.cpp
 *
 *  Created on: 02.04.2012
 *      Author: Thomas Wiemann
 */

#include "lvr2/display/InteractivePointCloud.hpp"

namespace lvr2
{

InteractivePointCloud::InteractivePointCloud()
{
	m_boundingBox = new BoundingBox<Vec>(
			Vec(-8, -8, -8),
			Vec(8, 8, 8)
			);

	updateBuffer(PointBufferPtr());
}

InteractivePointCloud::InteractivePointCloud(PointBufferPtr buffer)
{
	m_boundingBox = new BoundingBox<Vec>(
			Vec(-8, -8, -8),
			Vec(8, 8, 8)
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
		glDrawArrays(GL_POINTS, 0, m_buffer->numPoints());
	}
}

void InteractivePointCloud::updateBuffer(PointBufferPtr buffer)
{
	if(buffer)
	{
		if(!m_boundingBox)
		{
			m_boundingBox = new BoundingBox<Vec>;
			m_boundingBox->expand(Vec(8000, 8000, 8000));
		}

		size_t num_vertices = buffer->numPoints();
		float* vertices = buffer->getPointArray().get();

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

} /* namespace lvr2 */
