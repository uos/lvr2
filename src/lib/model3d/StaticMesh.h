/*
 * StaticMesh.h
 *
 *  Created on: 12.11.2008
 *      Author: Thomas Wiemann
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

	float* 			getVertices();
	unsigned int* 	getIndices();
	float*          getNormals();

	size_t			getNumberOfVertices();
	size_t			getNumberOfFaces();



private:
	void interpolateNormals();

protected:

	void compileDisplayList();
	void readPly(string filename);

	float* normals;
	float* vertices;
	float* colors;

	unsigned int* m_indices;

	bool finalized;

	size_t number_of_vertices;
	size_t number_of_faces;

};

void StaticMesh::render(){
	if(visible) {
		if(finalized && listIndex != -1){
			//glEnable(GL_LIGHTING);
//			glPolygonMode( GL_FRONT_AND_BACK, GL_LINE); //wire frame
			glShadeModel(GL_FLAT);  //disable color interpolation
			glPushMatrix();
			glMultMatrixf(transformation.getData());
			glCallList(listIndex);
			if(show_axes)
			{
				if(m_boundingBox) m_boundingBox->render();
				glCallList(axesListIndex);
			}
			glPopMatrix();
 		}
	}
}



#endif /* STATICMESH_H_ */
