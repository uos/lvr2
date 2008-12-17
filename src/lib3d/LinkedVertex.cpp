/*
 * TriangleVertex.cpp
 *
 *  Created on: 15.12.2008
 *      Author: twiemann
 */

#include "LinkedVertex.h"

TriangleVertex::TriangleVertex() {
	position = Vertex(0.0, 0.0, 0.0);
	normal = Normal(0.0, 0.0, 0.0);
	init();
}

TriangleVertex::TriangleVertex(Vertex pos){
	position = pos;
	normal = Normal(0.0, 0.0, 0.0);
	init();
}

TriangleVertex::TriangleVertex(Vertex pos, Normal nor){
	position = pos;
	normal = nor;
	init();
}

TriangleVertex::TriangleVertex(float pos[3]){
	position = Vertex(pos[0], pos[1], pos[2]);
	normal = Normal(0.0, 0.0, 0.0);
	init();
}

TriangleVertex::TriangleVertex(float pos[3], float nor[3]){
	position = Vertex(pos[0], pos[1], pos[2]);
	normal = Normal(nor[0], nor[1], nor[2]);
	init();
}

TriangleVertex::TriangleVertex(const TriangleVertex &other){
	position = other.position;
	normal = other.normal;
	active = other.active;
	cost = other.cost;
	min_cost_neighbor = other.min_cost_neighbor;
	index = other.index;
	summed_triangle_area = other.summed_triangle_area;

	for(int i = 0; i < 4; i++){
		for(int j = 0; j < 4; j++){
			quadric_cost[i][j] = other.quadric_cost[i][j];
		}
	}
}

TriangleVertex::~TriangleVertex() {
	vertex_neighbors.clear();
	triangle_neighbors.clear();
}

void TriangleVertex::init(){

	for(int i = 0; i < 4; i++){
		for(int j = 0; j < 4; j++){
			quadric_cost[i][j] = -1;
		}
	}
	active = false;
	cost = 0.0;
	min_cost_neighbor = -1;
	index = -1;
	summed_triangle_area = 0.0;
	closed = false;

}

void TriangleVertex::calcQuadric(LinkedTriangleMesh& m, bool use_area){
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			quadric_cost[i][j] = 0;
		}
	}

	set<int>::iterator pos;
	for (pos = triangle_neighbors.begin(); pos != triangle_neighbors.end(); ++pos)
	{
		int triIndex = *pos;
		LinkedTriangle t = m.getTriangle(triIndex);
		if (t.isActive())
		{
			float triArea = 1;
			if (use_area)
			{
				triArea = t.calculateArea();
				summed_triangle_area += triArea;
			}

			Normal normal = t.getNormal();

			const float a = normal.x;
			const float b = normal.y;
			const float c = normal.z;
			const float d = t.getD();

			// NOTE: we could optimize this a bit by calculating values
			// like a * b and then using that twice (for _Q[0][1] and _Q[1][0]),
			// etc., since the matrix is symmetrical.  For now, I don't think
			// it's worth it.
			quadric_cost[0][0] += triArea * a * a;
			quadric_cost[0][1] += triArea * a * b;
			quadric_cost[0][2] += triArea * a * c;
			quadric_cost[0][3] += triArea * a * d;

			quadric_cost[1][0] += triArea * b * a;
			quadric_cost[1][1] += triArea * b * b;
			quadric_cost[1][2] += triArea * b * c;
			quadric_cost[1][3] += triArea * b * d;

			quadric_cost[2][0] += triArea * c * a;
			quadric_cost[2][1] += triArea * c * b;
			quadric_cost[2][2] += triArea * c * c;
			quadric_cost[2][3] += triArea * c * d;

			quadric_cost[3][0] += triArea * d * a;
			quadric_cost[3][1] += triArea * d * b;
			quadric_cost[3][2] += triArea * d * c;
			quadric_cost[3][3] += triArea * d * d;
		}
	}
}

bool TriangleVertex::isBorder(LinkedTriangleMesh& m) {
	set<int>::iterator pos, pos2;
	for (pos = getVertexNeighbors().begin(); pos != getVertexNeighbors().end(); ++pos)
	{
		int triCount = 0;

		TriangleVertex& v = m.getVertex(*pos);

		for (pos2 = v.getTriangleNeighbors().begin(); pos2 != v.getTriangleNeighbors().end(); ++pos2)
		{
			if (m.getTriangle(*pos2).hasVertex(index) )
			{
				++triCount;
			}
		}

		if (1 == triCount)
		{
			return true;
		}
	}

	return false;
}


void TriangleVertex::getBorderEdges(set<Border> &borderSet, LinkedTriangleMesh& m){
	// So go through the list of all neighboring vertices, and see how many
	// triangles this vertex has in common w/ each neighboring vertex.  Normally
	// there will be two triangles in common, but if there is only one, then this
	// vertex is on an edge.
	set<int>::iterator pos, pos2;

	for (pos = getVertexNeighbors().begin(); pos != getVertexNeighbors().end(); ++pos)
	{
		int triCount = 0;
		int triIndex = -1;
		TriangleVertex& v = m.getVertex(*pos);
		for (pos2 = v.getTriangleNeighbors().begin(); pos2 != v.getTriangleNeighbors().end(); ++pos2)
		{
			if (m.getTriangle(*pos2).hasVertex(index) )
			{
				++triCount;
				triIndex = m.getTriangle(*pos2).getFaceIndex();
			}
		}

		if (1 == triCount) // if only one triangle in common, it's an edge
		{
			// store the smaller index first
			Border b;
			b.triIndex = triIndex;
			if (index < v.getIndex())
			{
				b.vert1 = index;
				b.vert2 = v.getIndex();
			}
			else
			{
				b.vert1 = v.getIndex();
				b.vert2 = index;
			}
			borderSet.insert(b);
		}
	}
}

TriangleVertex& TriangleVertex::operator=(const TriangleVertex& other){

	if(this == &other) return *this;

	position = other.position;
	normal = other.normal;
	active = other.active;
	cost = other.cost;
	min_cost_neighbor = other.min_cost_neighbor;
	index = other.index;
	summed_triangle_area = other.summed_triangle_area;
	closed = other.closed;

	for(int i = 0; i < 4; i++){
		for(int j = 0; j < 4; j++){
			quadric_cost[i][j] = other.quadric_cost[i][j];
		}
	}
	return *this;
}

TriangleVertex& TriangleVertex::operator=(const float p[3]){
	position = Vertex(p[0], p[1], p[2]);
	normal = Normal(0.0, 0.0, 0.0);
	triangle_neighbors.clear();
	vertex_neighbors.clear();
	init();
	return *this;
}

void TriangleVertex::getQuadric(float Qret[4][4])
{
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			Qret[i][j] = quadric_cost[i][j];
		}
	}
}

void TriangleVertex::setQuadric(float q[4][4]){
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			quadric_cost[i][j] = q[i][j];
		}
	}
}

bool TriangleVertex::operator==(const TriangleVertex& v) const{
	return (v.normal == normal && v.position == position);
}

bool TriangleVertex::operator!=(const TriangleVertex& v) const{
	return !(*this == v);
}

bool TriangleVertex::operator<(const TriangleVertex& o) const{
	return cost < o.cost;
}

bool TriangleVertex::operator>(const TriangleVertex& o) const{
	return cost > o.cost;
}


float TriangleVertex::operator[](const int i){
  float ret = 0.0;
  switch(i){
  case 0:
    ret = position.x;
    break;
  case 1:
    ret = position.y;
    break;
  case 2:
    ret = position.z;
    break;
  default:
    cout << "LinkedVertex: Warning: Access index out of range." << endl;

  }
  return ret;
}


