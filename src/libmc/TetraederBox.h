/*
 * TetraederBox.h
 *
 *  Created on: 07.10.2008
 *      Author: twiemann
 */

#ifndef TETRAEDERBOX_H_
#define TETRAEDERBOX_H_

#include <stdlib.h>
#include <iostream>

using namespace std;

#include "interpolator.h"
#include "Tetraeder.h"

#include "../mesh/colorVertex.h"
#include "../mesh/staticMesh.h"

class TetraederBox {
public:
	TetraederBox(Vertex v, float voxelsize);
	virtual ~TetraederBox();

	int getApproximation(int globalIndex, StaticMesh &mesh, Interpolator* df);

private:

	void calcCorners();
	void calcTetraeders();

	ColorVertex baseVertex;
	ColorVertex corners[8];
	Tetraeder tetraeders[6];

	float voxelsize;

};

#endif /* TETRAEDERBOX_H_ */
