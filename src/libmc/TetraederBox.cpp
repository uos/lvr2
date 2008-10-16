/*
 * TetraederBox.cpp
 *
 *  Created on: 07.10.2008
 *      Author: twiemann
 */

#include "TetraederBox.h"

TetraederBox::TetraederBox(Vertex v, float vs) {

	voxelsize = vs;
	baseVertex = ColorVertex(v, 0.0f, 1.0f, 0.0f);

}

TetraederBox::~TetraederBox() {

}

int TetraederBox::getApproximation(int globalIndex, StaticMesh &mesh, Interpolator* df){

	calcCorners();
	calcTetraeders();

	for(int i = 0; i < 6; i++) globalIndex = tetraeders[i].getApproximation(globalIndex, mesh, df);

	return globalIndex;

}

void TetraederBox::calcTetraeders(){

	Vertex vertices[4];

	vertices[0] = corners[4];
	vertices[1] = corners[1];
	vertices[2] = corners[0];
	vertices[3] = corners[3];
	tetraeders[0] = Tetraeder(vertices);

	vertices[0] = corners[4];
	vertices[1] = corners[1];
	vertices[2] = corners[2];
	vertices[3] = corners[3];
	tetraeders[1] = Tetraeder(vertices);

	vertices[0] = corners[4];
	vertices[1] = corners[7];
	vertices[2] = corners[2];
	vertices[3] = corners[3];
	tetraeders[2] = Tetraeder(vertices);

	vertices[0] = corners[4];
	vertices[1] = corners[2];
	vertices[2] = corners[5];
	vertices[3] = corners[1];
	tetraeders[3] = Tetraeder(vertices);

	vertices[0] = corners[4];
	vertices[1] = corners[2];
	vertices[2] = corners[5];
	vertices[3] = corners[7];
	tetraeders[4] = Tetraeder(vertices);

	vertices[0] = corners[6];
	vertices[1] = corners[2];
	vertices[2] = corners[5];
	vertices[3] = corners[7];
	tetraeders[5] = Tetraeder(vertices);
}

void TetraederBox::calcCorners(){

  uchar r, g, b;
  r = b = 0; g = 200;

  float vsh = 0.5 * voxelsize;

  corners[0] = ColorVertex(baseVertex.x - vsh,
					  baseVertex.y - vsh,
					  baseVertex.z - vsh,
					  r, g, b);

  corners[1] = ColorVertex(baseVertex.x + vsh,
					  baseVertex.y - vsh,
					  baseVertex.z - vsh,
					  r, g, b);

  corners[2] = ColorVertex(baseVertex.x + vsh,
					  baseVertex.y + vsh,
					  baseVertex.z - vsh,
					  r, g, b);

  corners[3] = ColorVertex(baseVertex.x - vsh,
					  baseVertex.y + vsh,
					  baseVertex.z - vsh,
					  r, g, b);

  corners[4] = ColorVertex(baseVertex.x - vsh,
					  baseVertex.y - vsh,
					  baseVertex.z + vsh,
					  r, g, b);

  corners[5] = ColorVertex(baseVertex.x + vsh,
					  baseVertex.y - vsh,
					  baseVertex.z + vsh,
					  r, g, b);

  corners[6] = ColorVertex(baseVertex.x + vsh,
					  baseVertex.y + vsh,
					  baseVertex.z + vsh,
					  r, g, b);

  corners[7] = ColorVertex(baseVertex.x - vsh,
					  baseVertex.y + vsh,
					  baseVertex.z + vsh,
					  r, g, b);
}
