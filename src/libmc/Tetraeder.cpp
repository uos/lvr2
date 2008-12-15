/*
 * Tetraeder.cpp
 *
 *  Created on: 07.10.2008
 *      Author: twiemann
 */

#include "Tetraeder.h"

Tetraeder::Tetraeder(){

}

Tetraeder::Tetraeder(Vertex v[]) {
	for(int i = 0; i < 4; i++) vertices[i] = ColorVertex(v[i].x, v[i].y, v[i].z, 0.0f, 1.0f, 0.0f);
}

void Tetraeder::initIntersections(){
	intersections[0] = calcIntersection( 0, 1);
	intersections[1] = calcIntersection( 0, 2);
	intersections[2] = calcIntersection( 0, 3);
	intersections[3] = calcIntersection( 1, 2);
	intersections[4] = calcIntersection( 2, 3);
	intersections[5] = calcIntersection( 1, 3);
}

ColorVertex Tetraeder::calcIntersection(int v1, int v2){

	float x = calcIntersection(vertices[v1].x, vertices[v2].x, values[v1], values[v2], true);
	float y = calcIntersection(vertices[v1].y, vertices[v2].y, values[v1], values[v2], true);
	float z = calcIntersection(vertices[v1].z, vertices[v2].z, values[v1], values[v2], true);

	return ColorVertex(x,y,z, 0.0f, 1.0f, 0.0f);
}


float Tetraeder::calcIntersection(float x1, float x2, float d1, float d2, bool interpolate){
	float intersection = x2 - d2 * (x1 - x2) / (d1 - d2);
	return intersection;
}

int Tetraeder::getApproximation(int globalIndex, TriangleMesh &mesh, Interpolator* df){

	for(int i = 0; i < 4; i++) values[i] = df->distance(vertices[i]);


	initIntersections();

	int index = 0;
	for(int i = 0; i < 4; i++) if(values[i] > 0) index |= (1 << i);

	int vertex_count = 0;
	int tmp_indices[6];
	Vertex tmp_vertices[6];

	for(int i = 0; TetraTable[index][i] != -1; i++){
		tmp_vertices[vertex_count] = intersections[TetraTable[index][i]];
		tmp_indices[vertex_count] = globalIndex;
		mesh.addVertex(intersections[TetraTable[index][i]]);
		mesh.addNormal(Normal(0.0, 0.0, 0.0));
		//mesh.addIndex(globalIndex);
		cout << "WARNING: TETRAEDER DOES NOT WORK" << endl;
		globalIndex++;
		vertex_count++;
	}

	Vertex diff1, diff2, normal;

	//Calculate surface normal
	for(int i = 0; i < vertex_count - 2; i+= 3){
		diff1 = tmp_vertices[i] - tmp_vertices[i+1];
		diff2 = tmp_vertices[i+1] - tmp_vertices[i+2];
		normal = diff1.cross(diff2);
		//Interpolate with normals in mesh
		for(int j = 0; j < 3; j++){
			mesh.interpolateNormal(normal, tmp_indices[i+j]);
		}
	}

	return globalIndex;
}

Tetraeder::~Tetraeder() {
	// TODO Auto-generated destructor stub
}
