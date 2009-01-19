/*
 * LinkedTriangleMesh.h
 *
 *  Created on: 15.12.2008
 *      Author: twiemann
 */

#ifndef LINKEDTRIANGLEMESH_H_
#define LINKEDTRIANGLEMESH_H_

#include <vector>
#include <iostream>

using std::vector;
using std::cout;
using std::endl;

#include "LinkedVertex.h"
#include "LinkedTriangle.h"
#include "BoundingBox.h"

#include "pmesh.h"

class LinkedTriangle;
class TriangleVertex;

class LinkedTriangleMesh : public TriangleMesh {
public:
	LinkedTriangleMesh();
	LinkedTriangleMesh(const LinkedTriangleMesh& other);

	TriangleVertex& getVertex(int index) {
		if(index >= (int)vertex_buffer.size()) cout << "Index out of range!" << endl << flush;
		return vertex_buffer[index];
	};

	LinkedTriangle& getTriangle(int index){return triangle_buffer[index];};

	virtual void addVertex(Vertex v);
	virtual void addTriangle(int v0, int v1, int v2);

	void closeHoles();
	void closeBorder();
	void rotateMesh();

	void setNumberOfVertices(int n){ num_verts = n;};
	void setNumberOfTriangles(int n){ num_triangles = n;};

	int getNumberOfVertices(){return num_verts;};
	int getNumberOfTriangles(){return num_triangles;};

	void normalize();

	void calcOneNormal(int vertex);
	void calcVertexNormals();

	void pmesh();
	int getIndex(vertex vert);

	virtual void finalize();

	virtual ~LinkedTriangleMesh();


private:

	void calcBoundingBox(BoundingBox& b);
	void calcAllQMatrices(LinkedTriangleMesh& m);

	vector<LinkedTriangle> triangle_buffer;
	vector<TriangleVertex>   vertex_buffer;

	int num_verts;
	int num_triangles;
};

#endif /* LINKEDTRIANGLEMESH_H_ */
