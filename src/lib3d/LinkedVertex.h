/*
 * TriangleVertex.h
 *
 *  Created on: 15.12.2008
 *      Author: Thomas Wiemann
 */

#ifndef LINKEDVERTEX_H_
#define LINKEDVERTEX_H_

#include <set>
#include <vector>

#include "BaseVertex.h"
#include "Normal.h"
#include "LinkedTriangle.h"
#include "LinkedTriangleMesh.h"

using namespace std;

class LinkedTriangleMesh;

struct Border{
	int vert1;
	int vert2;
	int triIndex;

	bool operator<(const Border& b) const{
		int v1, v2, b1, b2;

		//Make sure that the smallest index is alwas first
		if(vert1 < vert2){
			v1 = vert1;
			v2 = vert2;
		} else {
			v1 = vert2;
			v2 = vert1;
		}

		//The same with the border indices
		if(b.vert1 < b.vert2){
			b1 = b.vert1;
			b2 = b.vert2;
		} else {
			b1 = b.vert2;
			b2 = b.vert1;
		}

		if(v1 < b1) return true;
		if(v1 > b1) return false;

		return (v2 < b2);


	}
};

class TriangleVertex {
public:
	TriangleVertex();
	TriangleVertex(Vertex position);
	TriangleVertex(Vertex position, Normal normal);
	TriangleVertex(float position[3]);
	TriangleVertex(float position[3], float normal[3]);
	TriangleVertex(const TriangleVertex &other);

	virtual ~TriangleVertex();

	void addVertexNeighbor(int v){vertex_neighbors.insert(v);};
	void addTriangleNeighbor(int t){ triangle_neighbors.insert(t);};

	void removeVertexNeighbor(int v){ vertex_neighbors.erase(v);};
	void removeTriangleNeighbor(int t){triangle_neighbors.erase(t);};

	void setActive(bool a){active = a;};
	void setEdgeCost(float c){ cost = c;};
	void setMinCostNeighbor(int n) {min_cost_neighbor = n;};
	void setIndex(int i){index = i;};
	void setQuadric(float q[4][4]);
	void setNormal(Normal n){normal = n;};
	void setSummedTriangleArea(float area){summed_triangle_area = area;};
	void setClosed(bool c){ closed = c;};

	void calcQuadric(LinkedTriangleMesh& m, bool use_area);

	void getQuadric(float ret_quadric[4][4]);
	void getBorderEdges(set<Border> &borderSet, LinkedTriangleMesh& m);

	bool isActive(){return active;};
	bool isBorder(LinkedTriangleMesh& m);
	bool isClosed(){return closed;};

	set<int>& getVertexNeighbors() {return vertex_neighbors;};
	set<int>& getTriangleNeighbors() {return triangle_neighbors;};

	float getEdgeCost(){ return cost;};
	float getSummedTriangleArea(){ return summed_triangle_area;};

	int getMinCostNeighbor() const {return min_cost_neighbor;};
	int getIndex() const{return index;};

	Vertex getPosition(){return position;};

	///The vertex position
	mutable Vertex position;

	///The vertex normal
	mutable Normal normal;

	TriangleVertex& operator=(const TriangleVertex& v);
	TriangleVertex& operator=(const float position[3]);

	bool operator==(const TriangleVertex& v) const;
	bool operator!=(const TriangleVertex& v) const;

	bool operator<(const TriangleVertex&) const;
	bool operator>(const TriangleVertex&) const;

	float operator[](const int index);

	friend std::ostream& operator<<(std::ostream&, const TriangleVertex&);

private:

	///All vertices that are connected to this vertes via
	///an edge
	set<int> vertex_neighbors;

	///All triangles that contain this vertex
	set<int> triangle_neighbors;

	///False is vertex was removed
	bool active;

	///True if hole was closed
	bool closed;

	///Cost for removing this vertex
	float cost;

	///Vertex at the other end of the min cost edge
	int min_cost_neighbor;

	///Index of this vertex
	int index;

	///Quadric error cost
	float quadric_cost[4][4];

	///Summed area of surrounding triangles
	float summed_triangle_area;

	///Initialize quadric error matrix
	void init();

};

//std::ostream& operator<<(std::ostream& out, const TriangleVertex& t){
//	out << "Linked Triangle: " << endl;
//	out << t.position;
//	out << t.normal;
//	out << endl;
//	return out;
//}

#endif /* LINKEDVERTEX_H_ */
