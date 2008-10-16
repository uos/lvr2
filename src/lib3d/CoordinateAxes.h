/*
 * CoordinateAxes.h
 *
 *  Created on: 27.08.2008
 *      Author: twiemann
 */

#ifndef COORDINATEAXES_H_
#define COORDINATEAXES_H_

#include "Renderable.h"

class CoordinateAxes: public Renderable {
public:
	CoordinateAxes();
	CoordinateAxes(float);

	virtual ~CoordinateAxes();

	virtual void render();
	virtual void transform(Matrix4 m);

private:
	void drawArrow(float length, float radius, int nSubdivs = 12);
	void drawAxes(float length);
};

#endif /* COORDINATEAXES_H_ */
