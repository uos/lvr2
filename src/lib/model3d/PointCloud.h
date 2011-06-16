/*
 * PointCloud.h
 *
 *  Created on: 02.09.2008
 *      Author: twiemann
 */

#ifndef POINTCLOUD_H_
#define POINTCLOUD_H_

#include "Renderable.h"
#include "ColorVertex.h"

#include <vector>
#include <string>
#include <fstream>

using namespace std;

class PointCloud : public Renderable{
public:
    PointCloud(string filename);
    PointCloud();

    virtual ~PointCloud();
    virtual inline void render();

    vector<ColorVertex> getPoints(){return points;};
    void setPoints(){};

    void addPoint(float x, float y, float z, uchar r, uchar g, uchar b){
        m_boundingBox->expand(ColorVertex(x, y, z, r, g, b));
        points.push_back(ColorVertex(x, y, z, r, g, b));
    };

    void addPoint(const ColorVertex v) {
        m_boundingBox->expand(v);
        points.push_back(v);
    };
    void clear(){
        delete m_boundingBox;
        m_boundingBox = new BoundingBox;
        points.clear();
    };
    void initDisplayList();
//private:
    vector<ColorVertex> points;


private:
    int getFieldsPerLine(string filename);

};

inline void PointCloud::render(){
    glPushMatrix();
    glMultMatrixf(transformation.getData());
    if(show_axes) glCallList(axesListIndex);
    glDisable(GL_LIGHTING);
//    if(active){
//        glColor3f(1.0f, 0.0f, 0.0f);
//    } else {
//        glColor3f(0.0f, 0.9f, 0.0f);
//    }
    glPointSize(5.0);
    glBegin(GL_POINTS);
    for(size_t i = 0; i < points.size(); i++){
    	glColor3f( ( (float) points[i].r ) / 255, 
				( (float) points[i].g ) / 255, 
				( (float) points[i].b ) / 255 );
        glVertex3f(points[i].x,
                   points[i].y,
                   points[i].z);
    }
    
    glEnd();
    glPointSize(1.0);
    glEnable(GL_LIGHTING);
    glPopMatrix();
}

#endif /* POINTCLOUD_H_ */
