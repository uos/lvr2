/*
 * GroundPlane.h
 *
 *  Created on: 27.08.2008
 *      Author: Thomas Wiemann
 */

#ifndef GROUNDPLANE_H_
#define GROUNDPLANE_H_

#include "Renderable.hpp"

namespace lssr
{

class GroundPlane : public Renderable{

public:
	GroundPlane();
	GroundPlane(int increment, int count);

	virtual ~GroundPlane();
	virtual void render();
	virtual void transform(Matrix4<float> m);

private:
	void drawGrid(int increment, int count);

};

} // namespace lssr

#endif /* GROUNDPLANE_H_ */
