/*
 * LinkedTriangle.h
 *
 *  Created on: 15.12.2008
 *      Author: twiemann
 */

#ifndef LINKEDTRIANGLE_H_
#define LINKEDTRIANGLE_H_

#include "Triangle.h"
#include "LinkedVertex.h"
#include "LinkedTriangleMesh.h"

class TriangleVertex;
class LinkedTriangleMesh;

class LinkedTriangle : public Triangle{
public:
	LinkedTriangle();
	LinkedTriangle(int v0, int v1, int v2) : Triangle(v0, v1, v2){};
	LinkedTriangle(LinkedTriangleMesh* m, int v0, int v1, int v2) : Triangle(v0, v1, v2), mesh(m){};
	LinkedTriangle(const LinkedTriangle& triangle);

	void changeMesh(LinkedTriangleMesh* m){mesh = m;};
	void changeVertex(int from, int to);

	void getVertexIndices(int& v0, int& v1, int& v2);

	bool hasVertex(int v){return (v == v0 || v == v1 || v == v2);};

	virtual void calculateNormal();
	virtual void interpolateNormal();

	virtual float calculateArea();
	virtual float getD() const {return d;};

	virtual int getFaceIndex() const {return face_index;};
	virtual int getIndex(int n);

	virtual Normal getNormal(){ return normal;};
	virtual Vertex getVertex(int n);

	virtual ~LinkedTriangle(){};

private:
	LinkedTriangleMesh* mesh;
};

#endif /* LINKEDTRIANGLE_H_ */
