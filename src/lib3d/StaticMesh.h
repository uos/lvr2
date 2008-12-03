/*
 * StaticMesh.h
 *
 *  Created on: 12.11.2008
 *      Author: twiemann
 */

#ifndef STATICMESH_H_
#define STATICMESH_H_

#include <stdlib.h>
#include <string>
#include <string.h>

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

	void save(string filename);
	void load(string filename);

	virtual void finalize();

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
	if(finalized && listIndex != -1) glCallList(listIndex);
}



#endif /* STATICMESH_H_ */
