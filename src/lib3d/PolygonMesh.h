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

using namespace std;

#include "Renderable.h"
#include "Polygon.h"

class PolygonMesh : public Renderable{
public:
	PolygonMesh() : Renderable(){};
	PolygonMesh(string filename);
	virtual ~PolygonMesh();

	virtual void load(string filename);
	virtual void save(string filename);

	inline void render();

protected:
	void compileDisplayList();

private:
	vector<Polygon> polygons;

};

inline void PolygonMesh::render(){
	if(listIndex != -1 && active){
		glPushMatrix();
		glMultMatrixd(transformation.getData());
		glCallList(listIndex);
		if(show_axes) glCallList(axesListIndex);
		glPopMatrix();
	}
}

#endif /* POLYGONMESH_H_ */
