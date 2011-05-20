/*
 * Tube.h
 *
 *  Created on: 26.08.2008
 *      Author: Thomas Wiemann
 */

#ifndef TUBE_H_
#define TUBE_H_

#include "Quaternion.h"
#include "BaseVertex.h"
#include "Normal.h"
#include "Renderable.h"

class Tube : public Renderable {

public:
	Tube();
	Tube(const Tube& other);
	Tube(Vertex p1, Vertex p2, float radius);
	Tube(Matrix4, float r);

	virtual void render();

	~Tube();
	Vertex calcClosestPoint(Vertex query_point);

private:
	float radius;
	GLUquadricObj* quadric;
};

#endif /* TUBE_H_ */
