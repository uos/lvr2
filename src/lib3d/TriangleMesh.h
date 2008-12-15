/*
 * TriangleMesh.h
 *
 *  Created on: 17.10.2008
 *      Author: Thomas Wiemann
 */

#ifndef TRIANGLEMESH_H_
#define TRIANGLEMESH_H_

#include "Triangle.h"
#include "StaticMesh.h"

#include <vector>
#include <cassert>
using namespace std;

class Triangle;

class TriangleMesh : public StaticMesh{
public:

	TriangleMesh();
	TriangleMesh(const TriangleMesh &other);

	virtual void addTriangle(int v0, int v1, int v2);
	virtual void finalize();
	virtual ~TriangleMesh();

	void addVertex(Vertex v){ vertex_buffer.push_back(v);};
	void addNormal(Normal n){ normal_buffer.push_back(n);};

	void interpolateNormal(Normal n, size_t);
	void normalize();
	void printStats();

	Vertex   getVertex(int index);
	Normal   getNormal(int index);
	Triangle getTriangle(int index);

	TriangleMesh& operator=(const TriangleMesh&);

protected:

	vector<Normal> normal_buffer;
	vector<Vertex> vertex_buffer;

	vector<Triangle> triangle_buffer;

};

#endif /* TRIANGLEMESH_H_ */
