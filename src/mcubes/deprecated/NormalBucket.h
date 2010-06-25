/*
 * NormalBucket.h
 *
 *  Created on: 10.03.2009
 *      Author: twiemann
 */

#ifndef NORMALBUCKET_H_
#define NORMALBUCKET_H_

#include "../lib3d/BaseVertex.h"
#include "../lib3d/Normal.h"

#include <vector>

using namespace std;

class NormalBucket {
public:
	NormalBucket(): representative(Normal()), d(0.0f){};
	NormalBucket(Normal n, Vertex v);
	NormalBucket(const NormalBucket &o);

	bool insert(Normal n, Vertex v);

	virtual ~NormalBucket();

	Normal representative;
	float  d;

	vector<Normal> normals;
	vector<Vertex> vertices;

	const static float epsilon = 0.90;
};

#endif /* NORMALBUCKET_H_ */
