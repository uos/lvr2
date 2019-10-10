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
 * PointCloud.hpp
 *
 *  Created on: 20.08.2011
 *      Author: Thomas Wiemann
 */

#ifndef POINTCLOUD_H_
#define POINTCLOUD_H_

#include "lvr2/display/Renderable.hpp"

#include "lvr2/geometry/ColorVertex.hpp"

#include <vector>
#include <string>
#include <fstream>


using namespace std;


namespace lvr2
{

enum
{
    RenderPoints                = 0x01,
    RenderNormals               = 0x02,
};

class PointCloud : public Renderable{
public:
    using uColorVertex = ColorVertex<float, unsigned char>;
    PointCloud();
    PointCloud(ModelPtr loader, string name = "<unamed cloud>");
    PointCloud(PointBufferPtr buffer, string name = "<unamed cloud>");

    virtual ~PointCloud();
    virtual inline void render();

    vector<uColorVertex> getPoints(){return m_points;};
    void setPoints(){};

    void addPoint(float x, float y, float z, unsigned char r, unsigned char g, unsigned char b)
    {
        m_boundingBox->expand(Vec(x, y, z));
        m_points.push_back(uColorVertex(x, y, z, r, g, b));
    };

    void addPoint(const uColorVertex& v) 
    {
        m_boundingBox->expand(Vec(v.x, v.y, v.z));
        m_points.push_back(v);
    };

    void clear(){
        delete m_boundingBox;
        m_boundingBox = new BoundingBox<Vec>;
        m_points.clear();
    };

    void updateBuffer(PointBufferPtr buffer);

    void updateDisplayLists();
//private:
    vector<uColorVertex> m_points;

    void setRenderMode(int mode) {m_renderMode = mode;}


private:
    int getFieldsPerLine(string filename);
    void init(PointBufferPtr buffer);

    int                        m_renderMode;
    GLuint                     m_normalListIndex;
    floatArr                   m_normals;
    size_t                     m_numNormals;

};

inline void PointCloud::render()
{
    //cout << name << " : Active: " << " " << active << " selected : " << selected << endl;
    if(m_listIndex != -1 && m_active)
    {
        // Increase point size if normal rendering is enabled
        if(m_renderMode & RenderNormals)
        {
            glPointSize(5.0);
        }
        else
        {
            glPointSize(m_pointSize);
        }
        glDisable(GL_LIGHTING);
        glPushMatrix();
        glMultMatrixf(m_transformation.getData());

        // Render points
        if(m_selected)
        {
            glCallList(m_activeListIndex);
        }
        else
        {
            glCallList(m_listIndex);
        }

        // Render normals
        if(m_renderMode & RenderNormals)
        {
            glCallList(m_normalListIndex);
        }
        glPointSize(1.0);
        glEnable(GL_LIGHTING);
        glPopMatrix();
    }
}

} // namespace lvr2

#endif /* POINTCLOUD_H_ */
