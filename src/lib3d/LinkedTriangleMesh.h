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

#include "LinkedTriangle.h"

class LinkedTriangle;

class LinkedTriangleMesh : public TriangleMesh {
public:
	LinkedTriangleMesh();

	virtual void addTriangle(int v0, int v1, int v2);
	virtual void finalize();

	virtual ~LinkedTriangleMesh();

private:
	vector<LinkedTriangle> triangle_buffer;
};

#endif /* LINKEDTRIANGLEMESH_H_ */
