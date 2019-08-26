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
 * MultiPointCloud.h
 *
 *  @date 04.07.2011
 *  @author Thomas Wiemann
 */

#ifndef MULTIPOINTCLOUD_H_
#define MULTIPOINTCLOUD_H_

#include "lvr2/display/PointCloud.hpp"

#include <map>
#include <string>
#include <sstream>

using std::stringstream;
using std::map;
using std::string;

namespace lvr2
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

    using uColorVertex = ColorVertex<float, unsigned char>;
public:
    MultiPointCloud(ModelPtr model, string name = "<unnamed point cloud>");
    MultiPointCloud(PointBufferPtr buffer, string name = "<unnamed point cloud>");

    virtual ~MultiPointCloud()
    {
        for (auto p : m_clouds)
        {
            delete p.second->cloud;
            delete p.second;
        }
    }

    virtual inline void render();

    void addCloud(PointCloud* pc);
    void removeCloud(PointCloud* pc);

    pc_attr_it first() { return m_clouds.begin();}
    pc_attr_it last()  { return m_clouds.end();}

    //void exportAllPoints(string filename);

    virtual ModelPtr model();
private:

    void init(PointBufferPtr buffer);

    map<PointCloud*, PointCloudAttribute*> m_clouds;
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

} // namespace lvr2

#endif /* MULTIPOINTCLOUD_H_ */
