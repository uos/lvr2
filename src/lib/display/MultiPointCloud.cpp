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
using std::ofstream;

#include "MultiPointCloud.hpp"
#include <boost/filesystem.hpp>

#include "../io/UosIO.hpp"

namespace lssr
{

MultiPointCloud::MultiPointCloud(string dir)
{
    boost::filesystem::path directory(dir);
    if(is_directory(directory))
    {
        Model model;
        UosIO io;
        io.read( &model, dir );
        int n = io.getNumScans();
        for(int i = 0; i < n; i++)
        {
            indexPair p = io.getScanRange(i);

            // Create new point cloud from scan
            PointCloud* pc = new PointCloud;
            for(size_t a = p.first; a <= p.second; a++)
            {
                size_t n;
                float   ** points = model.m_pointCloud->getIndexedPointArray( n );
                uchar ** colors = model.m_pointCloud->getIndexedPointColorArray( n );
                if(colors)
                {
                    pc->addPoint(points[a][0], points[a][1], points[a][2], colors[a][0], colors[a][1], colors[a][2]);
                }
                else
                {
                    pc->addPoint(points[a][0], points[a][1], points[a][2], 255, 0, 0);
                }
            }
            pc->updateDisplayLists();
            pc->setName(dir);
            addCloud(pc);
        }
    }
    else
    {
        cout << "MultiPointCloud: " << dir << " is not a directory." << endl;
    }
}

MultiPointCloud::~MultiPointCloud()
{
    // TODO Auto-generated destructor stub
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

void MultiPointCloud::exportAllPoints(string filename)
{
    ofstream out(filename.c_str());
    if(out.good())
    {

        pc_attr_it it;
        for(it = m_clouds.begin(); it != m_clouds.end(); it++)
        {
            PointCloud* pc = it->second->cloud;
            if(pc->isActive())
            {
                cout << "Exporting points from " << pc->Name() << " to " << filename << endl;
                vector<uColorVertex>::iterator p_it;
                for(p_it = pc->m_points.begin(); p_it != pc->m_points.end(); p_it++)
                {
                    uColorVertex v = *p_it;
                    out << v.x << " " << v.y << " " << v.z <<  " "
                        << (int)v.r << " " << (int)v.g << " " << (int)v.b << endl;
                }
            }
        }
        out.close();
    }

}

}
