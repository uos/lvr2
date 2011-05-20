/*
 * ProgressiveMesh.h
 *
 *  Created on: 17.12.2008
 *      Author: Thomas Wiemann
 */

//#ifndef PROGRESSIVEMESH_H_
//#define PROGRESSIVEMESH_H_
//
//#include <ostream>
//#include <iostream>
//#include <vector>
//#include <list>
//#include <set>
//#include <map>
//
//#include <float.h>
//
//#include "LinkedTriangleMesh.h"
//
//using namespace std;
//
//struct EdgeCollapse{
//	int from;
//	int to;
//	set<int> removed_triangles;
//	set<int> affected_triangles;
//
//	EdgeCollapse(){
//		from = -1;
//		to = -1;
//		removed_triangles.clear();
//		affected_triangles.clear();
//	};
//
//	EdgeCollapse(const EdgeCollapse &o){
//		from = o.from;
//		to = o.to;
//		removed_triangles = o.removed_triangles;
//		affected_triangles = o.affected_triangles;
//	};
//
//};
//
//class VertexPointer{
//
//public:
//	VertexPointer(){
//		mesh = 0;
//		index = -1;
//	}
//
//	VertexPointer(const VertexPointer &other){
//		mesh = other.mesh;
//		index = other.index;
//	}
//
//	LinkedTriangleMesh* mesh;
//	int index;
//	bool operator<(const VertexPointer& vp) const{
//		return (mesh->getVertex(index) < mesh->getVertex(vp.index));
//	}
//};
//
//typedef multiset<VertexPointer, less<VertexPointer> > VertexPointerSet;
//
//class ProgressiveMesh {
//public:
//
//	enum EdgeCost{SHORTEST, MELAX, QUADRIC, QUADRICTRI, MAX_EDGECOST};
//
//	ProgressiveMesh(LinkedTriangleMesh* mesh, EdgeCost ec);
//
//	void collapseEdge();
//	void splitVertex();
//
//	void simplify(float reduction);
//
//	int getNumberOfCollapses(){ return edge_collapse_list.size();};
//	int getNumberOfEdgeCollapses(){ return edge_collapse_list.size();};
//	int getNumberOfTriangles(){ return reduced_mesh.getNumberOfTriangles();};
//	int getNumberOfVisitedTriangles(){ return number_of_visited_triangles;};
//
//	virtual void save(string filename);
//
//	virtual ~ProgressiveMesh(){};
//
//private:
//
//	const static int BOUNDARY_WEIGHT= 1000;
//
//	float shortEdgeCollapseCost(LinkedTriangleMesh& m, TriangleVertex &v);
//	float melaxCollapseCost(LinkedTriangleMesh& m, TriangleVertex &v);
//	float quadricCollapseCost(LinkedTriangleMesh& m, TriangleVertex &v);
//	float calcQuadricError(float quadric[4][4], TriangleVertex &v, float area);
//
//	void createEdgeCollapseList();
//	void calcAllQuadricMatrices(LinkedTriangleMesh& mesh, bool use_ares);
//
//	void calcEdgeCollapseCosts(
//			VertexPointerSet &vertexSet,
//			vector<VertexPointerSet::iterator> &vert_set_iterator,
//			int number_of_vertices, LinkedTriangleMesh &mesh,
//			EdgeCost &cost);
//
//	void calcQuadricMatrices(EdgeCost &cost, LinkedTriangleMesh &mesh);
//
//	void ensureEdgeCollapseIsValid(
//			EdgeCollapse &ec, TriangleVertex &vertex,
//			LinkedTriangleMesh &mesh, const EdgeCost &cost,
//			bool &bad_vertex);
//
//	void applyBorderPenalties(set<Border> &border_set, LinkedTriangleMesh &mesh);
//
//	void setToVertexQuadric(
//			TriangleVertex &to, TriangleVertex &from,
//			const EdgeCost &cost);
//
//	void updateTriangles(
//			EdgeCollapse &ec,
//			TriangleVertex &vc, set<int> &affectedVerts,
//			LinkedTriangleMesh &mesh);
//
//	void updateAffectedVertexNeighbors(
//			TriangleVertex &vertex, const EdgeCollapse &cost,
//			const set<int> &affected_vertices);
//
//	void updateAffectedVertices(
//			LinkedTriangleMesh& new_mesh,
//			vector<VertexPointerSet::iterator> &vertex_set_vector,
//			VertexPointerSet &vertex_set, const EdgeCollapse &edge_collapse,
//			set<int> &affected_vertices, const EdgeCost &cost,
//			set<int> &affected_quadric_vertices);
//
//	void resetAffectedVertexCosts(
//			const EdgeCost& cost, LinkedTriangleMesh &new_mesh,
//			TriangleVertex &vertex);
//
//	void removeVertexIfNecessary(
//			TriangleVertex &vert, VertexPointerSet &vertSet,
//			vector<VertexPointerSet::iterator> &vertSetVec,
//			LinkedTriangleMesh &mesh, const EdgeCost &cost,
//			set<int> &affectedQuadricVerts);
//
//	void recalcQuadricCollapseCosts(
//			set<int> &affected_quadric_vertices,
//			LinkedTriangleMesh &mesh, const EdgeCost &cost);
//
//	void buildEdgeCollapseList(
//			LinkedTriangleMesh &mesh, const EdgeCost &cost,
//			list<EdgeCollapse> & edge_collapse_list,
//			VertexPointerSet &vertex_set,
//			vector<VertexPointerSet::iterator> &vert_set_vector);
//
//	void calcMelaxMaxValue(
//			LinkedTriangleMesh &mesh, set<int> &adjacent_faces,
//			TriangleVertex &v, set<int> &neighbors,
//			float &max_value,
//			bool &max_value_found);
//
//
//	LinkedTriangleMesh* original_mesh;
//	LinkedTriangleMesh   reduced_mesh;
//
//	list<EdgeCollapse> edge_collapse_list;
//	list<EdgeCollapse>::iterator edge_collapse_it;
//
//	int number_of_visited_triangles;
//
//	EdgeCost edge_cost;
//
//	//Disable consty assignments....
//	ProgressiveMesh(const ProgressiveMesh& o);
//	ProgressiveMesh& operator=(const ProgressiveMesh&);
//	bool operator==(const ProgressiveMesh&);
//
//};
//
//#endif /* PROGRESSIVEMESH_H_ */
