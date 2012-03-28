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
 * MultiPointCloud.h
 *
 *  @date 04.07.2011
 *  @author Thomas Wiemann
 */

#ifndef MULTIPOINTCLOUD_H_
#define MULTIPOINTCLOUD_H_

#include "PointCloud.hpp"

#include <map>
#include <string>
#include <sstream>

using std::stringstream;
using std::map;
using std::string;

namespace lssr
{

struct PointCloudAttribute
{
    PointCloudAttribute() : cloud(0), marked(false), active(true) {}
    PointCloud*  cloud;
    bool         marked;
    bool         active;
};

typedef map<PointCloud*, PointCloudAttribute*> pc_attr_map;
typedef map<PointCloud*, PointCloudAttribute*>::iterator pc_attr_it;

class MultiPointCloud : public Renderable
{
public:
    MultiPointCloud(ModelPtr model, string name = "<unnamed point cloud>");
    MultiPointCloud(PointBufferPtr buffer, string name = "<unnamed point cloud>");

    virtual ~MultiPointCloud() {};
    virtual inline void render();

    void addCloud(PointCloud* pc);
    void removeCloud(PointCloud* pc);

    pc_attr_it first() { return m_clouds.begin();}
    pc_attr_it last()  { return m_clouds.end();}

    //void exportAllPoints(string filename);

    virtual ModelPtr model();
private:

    void init(PointBufferPtr buffer);

    map<PointCloud*, PointCloudAttribute*>    m_clouds;
};

void MultiPointCloud::render()
{
    if(!m_active) return;
    map<PointCloud*, PointCloudAttribute*>::iterator it;
    for(it = m_clouds.begin(); it != m_clouds.end(); it++)
    {
        it->second->cloud->render();
    }
}

}

#endif /* MULTIPOINTCLOUD_H_ */
