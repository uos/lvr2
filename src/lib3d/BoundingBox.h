/*
 * BoundingBox.h
 *
 *  Created on: 22.10.2008
 *      Author: twiemann
 */

#ifndef BOUNDINGBOX_H_
#define BOUNDINGBOX_H_

#include <math.h>

#include "Renderable.h"

class BoundingBox: public Renderable {
public:
	BoundingBox();
	BoundingBox(Vertex v1, Vertex v2);
	BoundingBox(float x_min, float y_min, float z_min,
			    float x_max, float y_max, float z_max);

	~BoundingBox();

	inline void expand(Vertex v);
	inline void expand(float x, float y, float z);
	inline void render();

	Vertex v_min;
	Vertex v_max;

	float  x_size;
	float  y_size;
	float  z_size;

};


inline void BoundingBox::expand(Vertex v){
	v_min.x = min(v.x, v_min.x);
	v_min.y = min(v.y, v_min.y);
	v_min.z = min(v.z, v_min.z);

	v_max.x = max(v.x, v_max.x);
	v_max.y = max(v.y, v_max.y);
	v_max.z = max(v.z, v_max.z);

	x_size = fabs(v_max.x - v_min.x);
	y_size = fabs(v_max.y - v_min.y);
	z_size = fabs(v_max.z - v_min.z);
}

inline void BoundingBox::expand(float x, float y, float z){
	v_min.x = min(x, v_min.x);
	v_min.y = min(y, v_min.y);
	v_min.z = min(z, v_min.z);

	v_max.x = max(x, v_max.x);
	v_max.y = max(y, v_max.y);
	v_max.z = max(z, v_max.z);

	x_size = fabs(v_max.x - v_min.x);
	y_size = fabs(v_max.y - v_min.y);
	z_size = fabs(v_max.z - v_min.z);
}

inline void BoundingBox::render(){

}


#endif /* BOUNDINGBOX_H_ */
