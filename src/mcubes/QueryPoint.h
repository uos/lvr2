/*
 * QueryPoint.h
 *
 *  Created on: 22.10.2008
 *      Author: Thomas Wiemann
 */

#ifndef QUERYPOINT_H_
#define QUERYPOINT_H_

#include "../model3d/BaseVertex.h"

class QueryPoint {
public:
	QueryPoint();
	QueryPoint(Vertex v);
	QueryPoint(Vertex v, float f);
	QueryPoint(const QueryPoint &o);

	virtual ~QueryPoint();

	Vertex position;
	float  distance;
};

#endif /* QUERYPOINT_H_ */
