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

 /**
 * MultiPointCloud.cpp
 *
 *  @date 04.07.2011
 *  @author Thomas Wiemann
 */

#include "lvr2/display/MultiPointCloud.hpp"

namespace lvr2
{

MultiPointCloud::MultiPointCloud(ModelPtr model, string name)
{

    PointBufferPtr buffer = model->m_pointCloud;
    init(buffer);
    m_model = model;
}

MultiPointCloud::MultiPointCloud(PointBufferPtr buffer, string name)
{
	m_model = ModelPtr(new Model(buffer));
	init(buffer);
}

void MultiPointCloud::init(PointBufferPtr buffer)
{
	if(buffer)
	{
        m_boundingBox = new BoundingBox<Vec>;
        size_t numSubClouds;
        size_t dummy; 

        indexArray subClouds = buffer->getIndexArray("sub_clouds", numSubClouds, dummy);

        vector<indexPair> pairs;
        pairs.reserve(numSubClouds);
        for (size_t i = 0; i < numSubClouds; i++)
        {
            pairs[i].first  = subClouds[i*2 + 0];    
            pairs[i].second = subClouds[i*2 + 1];    
        }        

		vector<indexPair>::iterator it;

		int c(1);
		size_t n = buffer->numPoints();
        size_t w_color;
		floatArr points = buffer->getPointArray();
		ucharArr colors = buffer->getColorArray(w_color);

		for(it = pairs.begin(); it != pairs.end(); it ++)
		{
			indexPair p = *it;

			// Create new point cloud from scan
			PointCloud* pc = new PointCloud;
			for(size_t a = p.first; a <= p.second; a++)
			{
				if(colors)
				{
					pc->addPoint(points[a*3 + 0], points[a*3 + 1], points[a*3 + 2], colors[a*3 + 0], colors[a*3 + 1], colors[a*3 + 2]);
				}
				else
				{
					pc->addPoint(points[a*3 + 0], points[a*3 + 1], points[a*3 + 2], 255, 0, 0);
				}
			}
			stringstream ss;

			pc->updateDisplayLists();
			pc->setName(ss.str());
			addCloud(pc);
			c++;
		}
	}
}

void MultiPointCloud::addCloud(PointCloud* pc)
{
    PointCloudAttribute* a = new PointCloudAttribute;
    a->cloud = pc;
    m_clouds[pc] = a;
    m_boundingBox->expand(*(pc->boundingBox()));
}

void MultiPointCloud::removeCloud(PointCloud* pc)
{
    m_clouds.erase(pc);
}

ModelPtr MultiPointCloud::model( )
{

    // Count all points that need to be exported
    pc_attr_it it;
    size_t c = 0;
    for(it = m_clouds.begin(); it != m_clouds.end(); it++)
    {
        PointCloud* pc = it->second->cloud;
        if(pc->isActive())
        {
            vector<uColorVertex>::iterator p_it;
            for(p_it = pc->m_points.begin(); p_it != pc->m_points.end(); p_it++)
            {
                c++;
            }
        }
    }


    // Create a new model and save points
    PointBufferPtr pcBuffer(new PointBuffer);
    floatArr pointBuffer(new float[3 * c]);
    ucharArr colorBuffer(new unsigned char[3 * c]);
    c = 0;

    for(it = m_clouds.begin(); it != m_clouds.end(); it++)
    {
        PointCloud* pc = it->second->cloud;
        if(pc->isActive())
        {
            vector<uColorVertex>::iterator p_it;
            for(p_it = pc->m_points.begin(); p_it != pc->m_points.end(); p_it++)
            {
                size_t bufferPos = 3 * c;

                uColorVertex v = *p_it;
                pointBuffer[bufferPos    ] = v.x;
                pointBuffer[bufferPos + 1] = v.y;
                pointBuffer[bufferPos + 2] = v.z;

                colorBuffer[bufferPos    ] = v.r;
                colorBuffer[bufferPos + 1] = v.g;
                colorBuffer[bufferPos + 2] = v.b;

                c++;
            }
        }

    }

    pcBuffer->setPointArray(pointBuffer, c);
    pcBuffer->setColorArray(colorBuffer, c);

    ModelPtr modelPtr(new Model);
    modelPtr->m_pointCloud = pcBuffer;


    return modelPtr;
}

} // namespace lvr2
