/*
 * FastBox.cpp
 *
 *  Created on: 22.10.2008
 *      Author: twiemann
 */

#include "FastBox.h"

FastBox::FastBox() {
	for(int i = 0; i < 8; i++){
		vertices[i]      = -1;
		configuration[i] = false;
	}

	for(int i = 0; i < 12; i++){
		intersections[i] = -1;
	}
}

FastBox::FastBox(const FastBox &other){
	for(int i = 0; i < 8; i++){
		vertices[i]      = other.vertices[i];
		configuration[i] = other.configuration[i];
	}

	for(int i = 0; i < 12; i++){
		intersections[i] = other.intersections[i];
	}
}

int FastBox::calcApproximation(vector<QueryPoint> &qp, TriangleMesh &mesh, int global_index){

	ColorVertex corners[8];
	ColorVertex vertex_positions[12];
	ColorVertex tmp_vertices[12];
	float distances[8];


	getCorners(corners, qp);
	getDistances(distances, qp);
	getIntersections(corners, distances, vertex_positions);

	int index = getIndex();
	int edge_index = 0;
	int vertex_count = 0;
	int tmp_indices[12];

	BaseVertex diff1, diff2;
	Normal normal;

	for(int a = 0; MCTable[index][a] != -1; a+= 3){
		for(int b = 0; b < 3; b++){

			edge_index = MCTable[index][a + b];

			mesh.addVertex(vertex_positions[edge_index]);
			mesh.addNormal(Normal(0.0, 0.0, 0.0));
			mesh.addIndex(global_index);


			//Count and tmp-save generated vertices
			tmp_indices[vertex_count] = global_index;
			tmp_vertices[vertex_count] = vertex_positions[edge_index];

			vertex_count++;
			global_index++;

			//Calculate surface normal
			for(int i = 0; i < vertex_count - 2; i+= 3){
				diff1 = tmp_vertices[i] - tmp_vertices[i+1];
				diff2 = tmp_vertices[i+1] - tmp_vertices[i+2];
				normal = diff1.cross(diff2);

				//Interpolate with normals in mesh
				for(int j = 0; j < 3; j++){
					mesh.interpolateNormal( normal, tmp_indices[i+j]);
				}
			}

		}

	}
	return global_index;
}



void FastBox::getCorners(ColorVertex corners[], vector<QueryPoint> &qp){

	for(int i = 0; i < 8; i++){
		corners[i] = ColorVertex(qp[vertices[i]].position, 0.0f, 1.0f, 0.0f);
	}

}
void FastBox::getDistances(float distances[], vector<QueryPoint> &qp){
	for(int i = 0; i < 8; i++){
		distances[i] = qp[vertices[i]].distance;
	}
}


int FastBox::getIndex() const{
  int index = 0;
  for(int i = 0; i < 8; i++){
    if(configuration[i] > 0) index |= (1 << i);
  }
  return index;
}

void FastBox::getIntersections(ColorVertex corners[], float distance[], ColorVertex positions[]){

	float current_color[3] = {0.0f, 1.0f, 0.0f};

	bool interpolate = true;
	float d1, d2;
	d1 = d2 = 0.0;

	float intersection;

	//Calc distances;
	for(int i = 0; i < 8; i++){
		if(distance[i] > 0) {
			configuration[i] = true;
		}
	}

	intersection = calcIntersection(corners[0].x, corners[1].x, distance[0], distance[1], interpolate);
	positions[0] = ColorVertex(intersection, corners[0].y, corners[0].z,
			current_color[0], current_color[1], current_color[2]);

	intersection = calcIntersection(corners[1].y, corners[2].y, distance[1], distance[2], interpolate);
	positions[1] = ColorVertex(corners[1].x, intersection, corners[1].z,
			current_color[0], current_color[1], current_color[2]);

	intersection = calcIntersection(corners[3].x, corners[2].x, distance[3], distance[2], interpolate);
	positions[2] = ColorVertex(intersection, corners[2].y, corners[2].z,
			current_color[0], current_color[1], current_color[2]);

	intersection = calcIntersection(corners[0].y, corners[3].y, distance[0], distance[3], interpolate);
	positions[3] = ColorVertex(corners[3].x, intersection, corners[3].z,
			current_color[0], current_color[1], current_color[2]);

	//Back Quad
	intersection = calcIntersection(corners[4].x, corners[5].x, distance[4], distance[5], interpolate);
	positions[4] = ColorVertex(intersection, corners[4].y, corners[4].z,
			current_color[0], current_color[1], current_color[2]);

	intersection = calcIntersection(corners[5].y, corners[6].y, distance[5], distance[6], interpolate);
	positions[5] = ColorVertex(corners[5].x, intersection, corners[5].z,
			current_color[0], current_color[1], current_color[2]);


	intersection = calcIntersection(corners[7].x, corners[6].x, distance[7], distance[6], interpolate);
	positions[6] = ColorVertex(intersection, corners[6].y, corners[6].z,
			current_color[0], current_color[1], current_color[2]);

	intersection = calcIntersection(corners[4].y, corners[7].y, distance[4], distance[7], interpolate);
	positions[7] = ColorVertex(corners[7].x, intersection, corners[7].z,
			current_color[0], current_color[1], current_color[2]);


	//Sides
	intersection = calcIntersection(corners[0].z, corners[4].z, distance[0], distance[4], interpolate);
	positions[8] = ColorVertex(corners[0].x, corners[0].y, intersection,
			current_color[0], current_color[1], current_color[2]);

	intersection = calcIntersection(corners[1].z, corners[5].z, distance[1], distance[5], interpolate);
	positions[9] = ColorVertex(corners[1].x, corners[1].y, intersection,
			current_color[0], current_color[1], current_color[2]);

	intersection = calcIntersection(corners[3].z, corners[7].z, distance[3], distance[7], interpolate);
	positions[10] = ColorVertex(corners[3].x, corners[3].y, intersection,
			current_color[0], current_color[1], current_color[2]);

	intersection = calcIntersection(corners[2].z, corners[6].z, distance[2], distance[6], interpolate);
	positions[11] = ColorVertex(corners[2].x, corners[2].y, intersection,
			current_color[0], current_color[1], current_color[2]);

}

float FastBox::calcIntersection(float x1, float x2, float d1, float d2, bool interpolate){
  float intersection = x2 - d2 * (x1 - x2) / (d1 - d2);
  return intersection;
}


FastBox::~FastBox() {
	// TODO Auto-generated destructor stub
}


