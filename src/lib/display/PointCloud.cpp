/*
 * PointCloud.cpp
 *
 *  Created on: 20.08.2011
 *      Author: Thomas Wiemann
 */

#include "PointCloud.hpp"
#include "io/AsciiIO.hpp"

#include <string.h>

namespace lssr
{

PointCloud::PointCloud()
{
}

PointCloud::PointCloud(string filename) : Renderable(filename) {


    m_boundingBox = new BoundingBox<Vertex<float> >;

    lssr::AsciiIO io;
    io.read(filename);

    float** p = io.getPointArray();
    size_t n = io.getNumPoints();
    unsigned char** c = io.getPointColorArray();

    for(size_t i = 0; i < n; i++)
    {
        float x = p[i][0];
        float y = p[i][1];
        float z = -p[i][2];

        unsigned char r, g, b;

        if(c)
        {
            r = c[i][0];
            g = c[i][1];
            b = c[i][2];
        }
        else
        {
            r = 0;
            g = 200;
            b = 0;
        }

        m_boundingBox->expand(x, y, z);
        points.push_back(uColorVertex(x, y, z, r, g, b));
    }

    updateDisplayLists();
}

void PointCloud::updateDisplayLists(){

    // Check for existing display list for normal rendering
    if(m_listIndex != -1) {
        cout<<"PointCloud::initDisplayList() delete display list"<<endl;
        glDeleteLists(m_listIndex,1);
    }

    // Create new display list and render points
    m_listIndex = glGenLists(1);
    glNewList(m_listIndex, GL_COMPILE);
    glBegin(GL_POINTS);

    for(size_t i = 0; i < points.size(); i++)
    {
        float r = points[i].r / 255.0;
        float g = points[i].g / 255.0;
        float b = points[i].b / 255.0;

        glColor3f(r, g, b);
        glVertex3f(points[i].x,
                   points[i].y,
                   points[i].z);
    }
    glEnd();
    glEndList();

    // Check for existing list index for rendering a selected point
    // cloud

    if(m_activeListIndex != -1)
    {
        cout<<"PointCloud::initDisplayList() delete  active display list"<<endl;
        glDeleteLists(m_activeListIndex,1);
    }

    m_activeListIndex = glGenLists(1);
    glNewList(m_activeListIndex, GL_COMPILE);
    glBegin(GL_POINTS);

    glColor3f(1.0, 1.0, 0.0);
    for(size_t i = 0; i < points.size(); i++)
    {

        glVertex3f(points[i].x,
                   points[i].y,
                   points[i].z);
    }
    glEnd();
    glEndList();




}

int PointCloud::getFieldsPerLine(string filename)
{
	ifstream in(filename.c_str());

	//Get first line from file
	char first_line[1024];
	in.getline(first_line, 1024);
	in.close();

	//Get number of blanks
	int c = 0;
	char* pch = strtok(first_line, " ");
	while(pch != NULL){
		c++;
		pch = strtok(NULL, " ");
	}

	in.close();

	return c;
}

PointCloud::~PointCloud() {
    // TODO Auto-generated destructor stub
}

}
