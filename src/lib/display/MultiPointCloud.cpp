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

#include <fstream>
#include <sstream>
using std::ofstream;
using std::stringstream;

#include "MultiPointCloud.hpp"
#include <boost/filesystem.hpp>

#include "../io/UosIO.hpp"

namespace lssr
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
		vector<indexPair> pairs = buffer->getSubClouds();
		vector<indexPair>::iterator it;

		int c(1);
		size_t n;
		coord3fArr points = buffer->getIndexedPointArray( n );
		color3bArr colors = buffer->getIndexedPointColorArray( n );

		for(it = pairs.begin(); it != pairs.end(); it ++)
		{
			indexPair p = *it;

			// Create new point cloud from scan
			PointCloud* pc = new PointCloud;
			for(size_t a = p.first; a <= p.second; a++)
			{
				if(colors)
				{
					pc->addPoint(points[a][0], points[a][1], points[a][2], colors[a][0], colors[a][1], colors[a][2]);
				}
				else
				{
					pc->addPoint(points[a][0], points[a][1], points[a][2], 255, 0, 0);
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
    PointBufferPtr pcBuffer( new PointBuffer);
    floatArr pointBuffer(new float[3 * c]);
    ucharArr colorBuffer(new uchar[3 * c]);
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
    pcBuffer->setPointColorArray(colorBuffer, c);

    ModelPtr modelPtr(new Model);
    modelPtr->m_pointCloud = pcBuffer;


    return modelPtr;

}

} // namespace lssr
