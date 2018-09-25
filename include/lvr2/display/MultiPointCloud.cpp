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
 * MultiPointCloud.cpp
 *
 *  @date 04.07.2011
 *  @author Thomas Wiemann
 */

#include <lvr2/display/MultiPointCloud.hpp>

namespace lvr2
{

MultiPointCloud::MultiPointCloud(ModelPtr model, string name)
{

    PointBuffer2Ptr buffer = model->m_pointCloud;
    init(buffer);
    m_model = model;
}

MultiPointCloud::MultiPointCloud(PointBuffer2Ptr buffer, string name)
{
	m_model = ModelPtr(new Model(buffer));
	init(buffer);
}

void MultiPointCloud::init(PointBuffer2Ptr buffer)
{
	if(buffer)
	{
        m_boundingBox = new lvr::BoundingBox<lvr::Vertex<float> >;
        size_t numSubClouds;
        unsigned dummy; 

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
        unsigned w_color;
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
            vector<lvr::uColorVertex>::iterator p_it;
            for(p_it = pc->m_points.begin(); p_it != pc->m_points.end(); p_it++)
            {
                c++;
            }
        }
    }


    // Create a new model and save points
    PointBuffer2Ptr pcBuffer( new PointBuffer2);
    floatArr pointBuffer(new float[3 * c]);
    ucharArr colorBuffer(new unsigned char[3 * c]);
    c = 0;

    for(it = m_clouds.begin(); it != m_clouds.end(); it++)
    {
        PointCloud* pc = it->second->cloud;
        if(pc->isActive())
        {
            vector<lvr::uColorVertex>::iterator p_it;
            for(p_it = pc->m_points.begin(); p_it != pc->m_points.end(); p_it++)
            {
                size_t bufferPos = 3 * c;

                lvr::uColorVertex v = *p_it;
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
