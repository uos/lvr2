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

#include "Renderable.hpp"
#include "io/MeshLoader.hpp"
#include "geometry/BoundingBox.hpp"

namespace lssr
{

class StaticMesh : public Renderable{
public:
	StaticMesh();
	StaticMesh(MeshLoader& loader, string name="<unnamed static mesh>");
	StaticMesh(const StaticMesh &o);
	~StaticMesh();
	inline void render();

	virtual void finalize();
	virtual void savePLY(string filename);

	float* 			getVertices();
	unsigned int* 	getIndices();
	float*          getNormals();

	size_t			getNumberOfVertices();
	size_t			getNumberOfFaces();



private:
	void interpolateNormals();
	void setDefaultColors();
	void calcBoundingBox();

protected:

	void compileDisplayList();
	void readPly(string filename);

	float* m_vertexNormals;
	float* m_faceNormals;
	float* m_vertices;
	float* m_colors;

	unsigned int* m_indices;

	bool m_finalized;

	size_t m_numVertices;
	size_t m_numFaces;

};

void StaticMesh::render(){
    cout << "RENDER 1" << endl;
	if(m_visible)
	{
	    cout << "RENDER 2" << endl;
		if(m_finalized && m_listIndex != -1){
		    cout << "RENDER 3" << endl;
			//glEnable(GL_LIGHTING);
//			glPolygonMode( GL_FRONT_AND_BACK, GL_LINE); //wire frame
			glShadeModel(GL_FLAT);  //disable color interpolation
			glPushMatrix();
			glMultMatrixf(m_transformation.getData());
			glCallList(m_listIndex);
			if(m_showAxes)
			{
				glCallList(m_axesListIndex);
			}
			glPopMatrix();
 		}
	}
}

} // namespace lssr

#endif /* STATICMESH_H_ */
