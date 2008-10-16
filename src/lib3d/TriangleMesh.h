/*
 * TriangleMesh.h
 *
 *  Created on: 13.10.2008
 *      Author: twiemann
 */

#ifndef TRIANGLEMESH_H_
#define TRIANGLEMESH_H_

#include <string.h>

#include <vector>
#include <fstream>
using namespace std;

#include "Renderable.h"
#include "ColorVertex.h"
#include "Normal.h"

#define PLY_LITTLE_ENDIAN "format binary_little_endian 1.0\n"
#define PLY_BIG_ENDIAN "format binary_big_endian 1.0\n"

struct PlyHeaderDescription{

  char ply[5] ;
  char format[50];
  char comment[256];

};

struct PlyVertexDescription{

  char element[15];
  unsigned int count;
  char property_x[20];
  char property_y[20];
  char property_z[20];
  char property_nx[20];
  char property_ny[20];
  char property_nz[20];
};

struct PlyFaceDescription{

  char face[5];
  unsigned int count;
  char property[40];

};

struct PlyVertex{
  double x;
  double y;
  double z;
  double nx;
  double ny;
  double nz;
  float r;
  float g;
  float b;

  double u;
  double v;

  int texture;
};

struct PlyFace{
  int vertexCount;
  int indices[3];
};

class TriangleMesh : public Renderable{
public:
	TriangleMesh(string filename);

	virtual inline void render();
	virtual ~TriangleMesh();

	inline int setColorMaterial(float r, float g, float b);

private:
	void   readPLY(string filename);
	void   initDisplayList();

	float* vertices;
	float* colors;
	float* normals;

	unsigned int*   indices;

	int    number_of_vertices;
	int    number_of_faces;

};

inline void TriangleMesh::render(){

	if(visible){
		glPushMatrix();
		glMultMatrixd(transformation.getData());
		if(show_axes) glCallList(axesListIndex);
		if(listIndex >= 0) glCallList(listIndex);
		glPopMatrix();
	}
}

inline void setColorMaterial(float r, float g, float b){

	float mat_specular[4];
	float mat_ambient[4];
	float mat_diffuse[4];

	float mat_shininess = 50;

	mat_specular[0] = 0.7f; mat_ambient[0]  = 0.5f * r; mat_diffuse[0]  = r;
	mat_specular[1] = 0.7f; mat_ambient[1]  = 0.5f * g; mat_diffuse[1]  = g;
	mat_specular[2] = 0.7f; mat_ambient[2]  = 0.5f * b; mat_diffuse[2]  = b;
	mat_specular[3] = 1.0f; mat_ambient[3]  = 1.0f; mat_diffuse[3]  = 1.0f;

	glMaterialfv(GL_FRONT, GL_SPECULAR,  mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, &mat_shininess);
	glMaterialfv(GL_FRONT, GL_AMBIENT,   mat_ambient);
	glMaterialfv(GL_FRONT, GL_DIFFUSE,   mat_diffuse);

}

#endif /* TRIANGLEMESH_H_ */
