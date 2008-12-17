/*
 * TriangleMesh.h
 *
 *  Created on: 17.10.2008
 *      Author: Thomas Wiemann
 */

#ifndef TRIANGLEMESH_H_
#define TRIANGLEMESH_H_

#include "StaticMesh.h"

#include <vector>
#include <cassert>
using namespace std;

class TriangleMesh : public StaticMesh{
public:

	TriangleMesh();

	void addIndex(int index){ index_buffer.push_back(index);};
	void addVertex(Vertex v){ vertex_buffer.push_back(v);};
	void addNormal(Normal n){ normal_buffer.push_back(n);};

	Vertex getVertex(int n);

	void interpolateNormal(Normal n, size_t);
	void finalize();

	void printStats();

	virtual ~TriangleMesh();

private:

	vector<Normal> normal_buffer;
	vector<Vertex> vertex_buffer;
	vector<int>    index_buffer;

};

#endif /* TRIANGLEMESH_H_ */
