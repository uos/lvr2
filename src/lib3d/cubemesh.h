// ******************************
// mesh.h
//
// Mesh class, which stores a list
// of vertices & a list of triangles.
//
// Jeff Somers
// Copyright (c) 2002
//
// jsomers@alumni.williams.edu
// March 27, 2002
// ******************************

#ifndef __cubemesh_h
#define __cubemesh_h

#if defined (_MSC_VER) && (_MSC_VER >= 1020)

#pragma once
#pragma warning(disable:4710) // function not inlined
#pragma warning(disable:4702) // unreachable code
#pragma warning(disable:4514) // unreferenced inline function has been removed
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#endif

#include <fstream>
#include <iostream>
using std::cout;
using std::cerr;
using std::endl;
using std::ios;
using std::ofstream;
using std::ifstream;

using namespace std;

#include <stdlib.h>
#include <string.h>
#include <vector>
#include "vertex.h"
#include "vec3.h"
#include "triangle.h"

// Mesh class.  This stores a list of vertices &
// another list of triangles (which references the vertex list)
class CubeMesh
{
public:
	// Constructors and Destructors
	//Mesh() {_numVerts = _numTriangles = 0;};
	CubeMesh(); // passed name of mesh file
	~CubeMesh();

	CubeMesh(const CubeMesh&); // copy ctor
	CubeMesh& operator=(const CubeMesh&); // assignment op

	// Get list of vertices, triangles
	vertex& getVertex(int index) {return _vlist[index];};
	const vertex& getVertex(int index) const {return _vlist[index];};
	triangle& getTri(int index) {return _plist[index];};
	const triangle& getTri(int index) const {return _plist[index];};

     bool addVertex(float x,float y,float z);
	int getNumVerts() {return _numVerts;};
	void setNumVerts(int n) {_numVerts = n;};
  
     bool addTriangle(int v1,int v2,int v3);
     bool closeHoles(int number);
     bool rotateMesh(float x,float y, float z);
     bool closeBorder(vertex vert, int lastindex, int start);
	int getNumTriangles() {return _numTriangles;};
	void setNumTriangles(int n) {_numTriangles = n;};

	void Normalize();// center mesh around the origin & shrink to fit in [-1, 1]

	void calcOneVertNormal(unsigned vert); // recalc normal for one vertex

	void dump(); // print mesh state to cout
	void calcVertNormals(); // Calculate the vertex normals after loading the mesh
     bool toDXF(char *filename);

private:
	vector<vertex> _vlist; // list of vertices in mesh
	vector<triangle> _plist; // list of triangles in mesh

	int _numVerts;
	int _numTriangles;

	bool operator==(const CubeMesh&); // don't allow op== -- too expensive
	
	bool loadFromFile(char* filename); // load from PLY file

	void ChangeStrToLower(char* pszUpper)
	{
		for(char* pc = pszUpper; pc < pszUpper + strlen(pszUpper); pc++) {
			*pc = (char)tolower(*pc);
		}
	}

	// get bounding box for mesh
	void setMinMax(float min[3], float max[3]);

	void calcAllQMatrices(CubeMesh& mesh); // used for Quadrics method


  // Helper function for reading PLY mesh file
  //bool readNumPlyVerts(FILE *&inFile, int& nVerts);
  //bool readNumPlyTris(FILE *&inFile, int& nTris);
  //bool readPlyHeader(FILE *&inFile);
  //bool readPlyVerts(FILE *&inFile);
  //bool readPlyTris(FILE *&inFile);
};

#endif // __cubemesh_h

