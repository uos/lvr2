/*
 * PolygonMesh.h
 *
 *  Created on: 03.02.2009
 *      Author: twiemann
 */

#ifndef POLYGONMESH_H_
#define POLYGONMESH_H_

#include <fstream>
#include <iostream>
#include <cstring>
#include <list>

using namespace std;

#include "Renderable.h"
#include "Polygon.h"

using namespace Lib3D;

void vertexCallback(GLvoid *vertex);

bool comparePolygons(Polygon a, Polygon b);

class PolygonMesh : public Renderable{
public:
    PolygonMesh() : Renderable(){
        //drawFilled = false;
    };
    PolygonMesh(string filename);
    virtual ~PolygonMesh();

    virtual void load(string filename);
    virtual void save(string filename);

    inline void addPolygon(Polygon p){
    	polygons.push_back(p);
    	vector<Vertex>::iterator it;
    	for(it = p.vertices.begin(); it != p.vertices.end(); it++)
    	{
    		m_boundingBox->expand(*it);
    	}
    };
    
    inline void clear(){
//    	cout << "clearPolygon" << endl;
    	polygons.clear();
    };

    inline void sort() {
    	polygons.sort(comparePolygons);
    }    
    inline void render();
    //bool drawFilled;
protected:
    void compileDisplayList();

private:
    list<Polygon> polygons;
};

inline void PolygonMesh::render(){

	Polygon p;
	glPushAttrib(GL_ALL_ATTRIB_BITS);
	glEnable(GL_BLEND);
	glBlendFunc(GL_DST_COLOR, GL_SRC_ALPHA);
	//glColor3f(0.0, 0.0, 0.0);

	list<Polygon>::iterator i;
	for(i=polygons.begin(); i != polygons.end(); ++i) {
		p = *i;
		glColor4f(p.color_r, p.color_g, p.color_b, 0.7);
		glBegin(GL_POLYGON);
		for(size_t j = 0; j < p.vertices.size(); j++){
			glVertex3f(p.vertices[j].x, p.vertices[j].y, p.vertices[j].z);
		}
		glEnd();
	}
	glDisable(GL_BLEND);
	glPopAttrib();
}

#endif /* POLYGONMESH_H_ */
