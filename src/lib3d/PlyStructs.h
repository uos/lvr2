/*
 * PlyStructs.h
 *
 *  Created on: 21.10.2008
 *      Author: twiemann
 */

#ifndef PLYSTRUCTS_H_
#define PLYSTRUCTS_H_

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


#endif /* PLYSTRUCTS_H_ */
