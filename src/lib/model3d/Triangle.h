/*
 * Triangle.h
 *
 *  Created on: 11.12.2008
 *      Author: twiemann
 */

#ifndef TRIANGLE_H_
#define TRIANGLE_H_

#include <cassert>
#include <iostream>

#include "BaseVertex.h"
#include "Normal.h"
#include "TriangleMesh.h"

class TriangleMesh;

class Triangle {
public:
	Triangle();
	Triangle(const Triangle &o);
	Triangle(TriangleMesh* mesh, int v0, int v1, int v2);
	Triangle(int v0, int v1, int v2);

	virtual ~Triangle();

	Triangle& operator=(const Triangle &t){
		if(&t == this) return *this;
		v0              = t.v0;
		v1              = t.v1;
		v2              = t.v2;

		face_index      = t.face_index;
		normal          = t.normal;
		normal_array[0] = t.normal_array[0];
		normal_array[1] = t.normal_array[1];
		normal_array[2] = t.normal_array[2];
		mesh            = t.mesh;

		return *this;
	}

	bool operator==(const Triangle& t);
	bool operator!=(const Triangle& t);

	void setActive(bool _active) {active = _active;};
	void setFaceIndex(int index){face_index = index;};

	bool isActive() const{return active;};

	virtual void calculateNormal();
	virtual void interpolateNormal();

	virtual float calculateArea();
	virtual float getD() const {return d;};

	virtual int getFaceIndex() const {return face_index;};
	virtual int getIndex(int n);

	virtual Normal getNormal(){ return normal;};
	virtual Vertex getVertex(int n);

	friend std::ostream& operator<<(std::ostream& os, const Triangle& t);
protected:

	int    v0;
	int    v1;
	int    v2;
	int    face_index;

	Normal normal;
	Normal normal_array[3];

	float  d;
	bool   active;

	TriangleMesh* mesh;
};




#endif /* TRIANGLE_H_ */
