/*
 * StaticMesh.h
 *
 *  Created on: 12.11.2008
 *      Author: twiemann
 */

#ifndef STATICMESH_H_
#define STATICMESH_H_

#ifdef WIN32
#pragma warning(disable: 4996)
#endif

#include <stdlib.h>
#include <string>
#include <string.h>
#include <sstream>

using namespace std;

#include "Renderable.h"
#include "PlyStructs.h"

class StaticMesh : public Renderable{
public:
	StaticMesh();
	StaticMesh(string name);
	StaticMesh(const StaticMesh &o);
	~StaticMesh();
	inline void render();

	virtual void save(string filename);
	virtual void load(string filename);

	virtual void finalize();

	virtual void savePLY(string filename);

protected:

	void compileDisplayList();
	void readPly(string filename);

	float* normals;
	float* vertices;
	float* colors;

	unsigned int* indices;

	bool finalized;

	int number_of_vertices;
	int number_of_faces;
};

void StaticMesh::render(){
	if(finalized && listIndex != -1){
		glPushMatrix();
		glMultMatrixf(transformation.getData());
		glCallList(listIndex);
		if(show_axes) glCallList(axesListIndex);
		glPopMatrix();
	}
}



#endif /* STATICMESH_H_ */
