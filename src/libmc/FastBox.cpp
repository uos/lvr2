/*
 * FastBox.cpp
 *
 *  Created on: 22.10.2008
 *      Author: twiemann
 */

#include "FastBox.h"

int FastBox::neighbor_table[12][3] = {
  {12, 10,  9},
  {22, 12, 21},
  {16, 12, 15},
  { 4,  3, 12},
  {14, 10, 11},
  {23, 22, 14},
  {14, 16, 17},
  { 4,  5, 14},
  { 4,  1, 10},
  {22, 19, 10},
  { 4,  7, 16},
  {22, 25, 16}
};

int FastBox::neighbor_vertex_table[12][3] = {
  { 4,  2,  6},
  { 3,  5,  7},
  { 0,  6,  4},
  { 1,  5,  7},
  { 0,  6,  2},
  { 3,  7,  1},
  { 2,  4,  0},
  { 5,  1,  3},
  { 9, 11, 10},
  { 8, 10, 11},
  {11,  9,  8},
  {10,  8,  9}
};

FastBox::FastBox() {

	for(int i = 0; i < 8; i++){
		vertices[i]      = -1;
		configuration[i] = false;
	}

	for(int i = 0; i < 12; i++){
		intersections[i] = -1;
	}

	for(int i = 0; i < 27; i++){
		neighbors[i] = 0;
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

	for(int i = 0; i < 27; i++){
		neighbors[i] = other.neighbors[i];
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

	int current_index = 0;
	int triangle_indices[3];

	for(int a = 0; MCTable[index][a] != -1; a+= 3){
		for(int b = 0; b < 3; b++){
			edge_index = MCTable[index][a + b];
			current_index = -1;

			//If current vertex index doesn't exist
			//look for it in the suitable neighbor boxes
			if(intersections[edge_index] == -1){
				for(int i = 0; i < 3; i++){
					FastBox* current_neighbor = neighbors[neighbor_table[edge_index][i]];

					//If neighbor exists search for suitable index
					if(current_neighbor != 0){
						if(current_neighbor->intersections[neighbor_vertex_table[edge_index][i]] != -1){
							current_index = current_neighbor->intersections[neighbor_vertex_table[edge_index][i]];
						}
					}
				}
			}

			//If no index was found generate new index and vertex
			//and update all neighbor boxes
			if(current_index == -1){
				intersections[edge_index] = global_index;
				ColorVertex v = vertex_positions[edge_index];
				mesh.addVertex(v);
				mesh.addNormal(Normal());
				for(int i = 0; i < 3; i++){
					FastBox* current_neighbor = neighbors[neighbor_table[edge_index][i]];
					if(current_neighbor != 0){
						current_neighbor->intersections[neighbor_vertex_table[edge_index][i]] = global_index;
					}
				}
				global_index++;
			} else {
				intersections[edge_index] = current_index;
			}

			//Save vertices and indices for normal calculation
			tmp_vertices[vertex_count] = vertex_positions[edge_index];
			tmp_indices[vertex_count]  = intersections[edge_index];

			//Save vertex index in mesh
			//mesh.addIndex(intersections[edge_index]);
			triangle_indices[b] = intersections[edge_index];
			//Count generated vertices
			vertex_count++;
		}
		mesh.addTriangle(triangle_indices[0], triangle_indices[1], triangle_indices[2]);
	}

	//Calculate normals
	for(int i = 0; i < vertex_count - 2; i+= 3){
		diff1 = tmp_vertices[i] - tmp_vertices[i+1];
		diff2 = tmp_vertices[i+1] - tmp_vertices[i+2];
		normal = diff1.cross(diff2);

		//Interpolate with normals in mesh
		for(int j = 0; j < 3; j++){
			mesh.interpolateNormal( normal, tmp_indices[i+j]);
		}
	}

	return global_index;
}


int FastBox::calcApproximation(vector<QueryPoint> &qp, HalfEdgeMesh &mesh, int global_index){

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

	int current_index = 0;

	HalfEdgeVertex* vertex = 0;
	HalfEdgeFace* face     = 0;
	HalfEdge* edges[3];

	for(int a = 0; MCTable[index][a] != -1; a+= 3){
		face = new HalfEdgeFace;
		face->used = false;
		edges[0] = edges[1] = edges[2];

		for(int b = 0; b < 3; b++){
			edge_index = MCTable[index][a+b];
			current_index = -1;
			if(intersections[edge_index] == -1){
				for(int i = 0; i < 3; i++){
					FastBox* current_neighbor = neighbors[neighbor_table[edge_index][i]];
					if(current_neighbor != 0){
						if(current_neighbor->intersections[neighbor_vertex_table[edge_index][i]] != -1){
							current_index = current_neighbor->intersections[neighbor_vertex_table[edge_index][i]];
						}
					}
				}

				if(current_index == -1){
					intersections[edge_index] = global_index;
					vertex = new HalfEdgeVertex;
					vertex->position = vertex_positions[edge_index];
					vertex->index = global_index;

					mesh.he_vertices.push_back(vertex);
					face->index[b] = global_index;

					for(int i = 0; i < 3; i++){
						FastBox* current_neighbor = neighbors[neighbor_table[edge_index][i]];
						if(current_neighbor != 0){
							current_neighbor->intersections[neighbor_vertex_table[edge_index][i]] = global_index;
						}
					}
					global_index++;
				} else {
					intersections[edge_index] = current_index;
					face->index[b] = current_index;
				}
			}else {
				face->index[b] = intersections[edge_index];
			}


			//Save vertices and indices for normal calculation
			tmp_vertices[vertex_count] = vertex_positions[edge_index];
			tmp_indices[vertex_count]  = intersections[edge_index];

			//Count generated vertices
			vertex_count++;
		}


		for(int k = 0; k < 3; k++){
			HalfEdgeVertex* current = mesh.he_vertices[face->index[k]];
			HalfEdgeVertex* next    = mesh.he_vertices[face->index[(k+1) % 3]];

			HalfEdge* edgeToVertex = halfEdgeToVertex(current, next);

			if(edgeToVertex != 0){
				edges[k] = edgeToVertex->pair;
				edges[k]->face = face;
			} else {
				HalfEdge* edge = new HalfEdge;
				edge->face = face;
				edge->start = current;
				edge->end = next;

				HalfEdge* pair = new HalfEdge;
				pair->start = next;
				pair->end = current;
				pair->face = 0;

				edge->pair = pair;
				pair->pair = edge;

				current->out.push_back(edge);
				next->in.push_back(edge);

				current->in.push_back(pair);
				next->out.push_back(pair);

				edges[k] = edge;
			}

		}

		for(int k = 0; k < 3; k++){
			edges[k]->next = edges[(k+1) % 3];
		}

		//cout << ":: " << face->index[0] << " " << face->index[1] << " " << face->index[2] << endl;

		face->edge = edges[0];
		face->calc_normal();
		face->mcIndex = index;
		mesh.he_faces.push_back(face);
		face->face_index = mesh.he_faces.size();

	}

	//Calculate normals
	for(int i = 0; i < vertex_count - 2; i+= 3){
		diff1 = tmp_vertices[i] - tmp_vertices[i+1];
		diff2 = tmp_vertices[i+1] - tmp_vertices[i+2];
		normal = diff1.cross(diff2);

		//Interpolate with normals in mesh
		for(int j = 0; j < 3; j++){
			mesh.he_vertices[tmp_indices[i+j]]->normal += normal;
		}
	}


	return global_index;
}


HalfEdge* FastBox::halfEdgeToVertex(HalfEdgeVertex* v, HalfEdgeVertex* next){

  HalfEdge* edge = 0;
  HalfEdge* cur = 0;

  vector<HalfEdge*>::iterator it;

  for(it = v->in.begin(); it != v->in.end(); it++){

    cur = *it;
    if(cur->end == v && cur->start == next){
	 edge = cur;
    }

  }

  return edge;

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


