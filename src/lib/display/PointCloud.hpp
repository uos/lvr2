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
 * PointCloud.hpp
 *
 *  Created on: 20.08.2011
 *      Author: Thomas Wiemann
 */

#ifndef POINTCLOUD_H_
#define POINTCLOUD_H_

#include "Renderable.hpp"

#include "geometry/ColorVertex.hpp"

#include <vector>
#include <string>
#include <fstream>

using namespace std;

namespace lssr
{

class PointCloud : public Renderable{
public:

    PointCloud();
    PointCloud(ModelPtr loader, string name = "<unamed cloud>");

    virtual ~PointCloud();
    virtual inline void render();

    vector<uColorVertex> getPoints(){return m_points;};
    void setPoints(){};

    void addPoint(float x, float y, float z, uchar r, uchar g, uchar b){
        m_boundingBox->expand(uColorVertex(x, y, z, r, g, b));
        m_points.push_back(uColorVertex(x, y, z, r, g, b));
    };

    void addPoint(const uColorVertex v) {
        m_boundingBox->expand(v);
        m_points.push_back(v);
    };

    void clear(){
        delete m_boundingBox;
        m_boundingBox = new BoundingBox<Vertex<float> >;
        m_points.clear();
    };

    void updateDisplayLists();
//private:
    vector<uColorVertex> m_points;


private:
    int getFieldsPerLine(string filename);

};

inline void PointCloud::render()
{
    //cout << name << " : Active: " << " " << active << " selected : " << selected << endl;
    if(m_listIndex != -1 && m_active)
    {
        glDisable(GL_LIGHTING);
        glPushMatrix();
        glMultMatrixf(m_transformation.getData());

        if(m_selected)
        {
            glCallList(m_activeListIndex);
        }
        else
        {
            glCallList(m_listIndex);
        }
        glEnable(GL_LIGHTING);
        glPopMatrix();
    }
}

} // namespace lssr

#endif /* POINTCLOUD_H_ */
