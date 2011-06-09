/*
 * HalfEdgeMesh.h
 *
 *  Created on: 13.11.2008
 *      Author: Thomas Wiemann
 */

#ifndef HALFEDGEMESH_H_
#define HALFEDGEMESH_H_

#ifdef _MSC_VER
#include <hash_map>
using stdext::hash_map;
#else
#include <ext/hash_map>
using __gnu_cxx::hash_map;
#endif

#include <vector>
#include <stack>
#include <set>
#include <list>

using namespace std;

#include "Vertex.hpp"
#include "Normal.hpp"
#include "BaseMesh.hpp"

#include "HalfEdgeVertex.hpp"
#include "HalfEdge.hpp"
#include "HalfEdgeFace.hpp"

//#include "HalfEdgePolygon.h"

namespace lssr
{

template<typename VertexT, typename NormalT>
class HalfEdgeVertex;

template<typename VertexT, typename NormalT>
class HalfEdgeFace;

/**
 * @brief A implementation of a half edge triangle mesh.
 */
template<typename VertexT, typename NormalT>
class HalfEdgeMesh : public BaseMesh<VertexT, NormalT>
{
public:
	typedef HalfEdgeFace<VertexT, NormalT> HFace;
	typedef HalfEdgeVertex<VertexT, NormalT> HVertex;
	typedef HalfEdge<HVertex, HFace> HEdge;

	/**
	 * @brief   Ctor.
	 */
	HalfEdgeMesh();

	/**
	 * @brief   Dtor.
	 */
	virtual ~HalfEdgeMesh() {};

	/**
	 * @brief 	This method should be called every time
	 * 			a new vertex is created.
	 *
	 * @param	v 		A supported vertex type. All used vertex types
	 * 					must support []-access.
	 */
	virtual void addVertex(VertexT v);

	/**
	 * @brief 	This method should be called every time
	 * 			a new vertex is created to ensure that vertex
	 * 			and normal buffer always have the same size
	 *
	 * @param	n 		A supported vertex type. All used vertex types
	 * 					must support []-access.
	 */
	virtual void addNormal(NormalT n);

	/**
	 * @brief 	Insert a new triangle into the mesh
	 *
	 * @param	a 		The first vertex of the triangle
	 * @param 	b		The second vertex of the triangle
	 * @param	c		The third vertex of the triangle
	 */
	virtual void addTriangle(uint a, uint b, uint c);


	/**
	 * @brief	Flip the edge between f1 and f2
	 *
	 * @param	f1		The first face
	 * @param	f2		The second face
	 */
	virtual void flipEdge(HFace* f1, HFace* f2);

	/**
	 * @brief	Flip the given edge
	 *
	 * @param	edge	The edge to flip
	 */
	virtual void flipEdge(HEdge* edge);

	/**
	 * @brief 	Finalizes a mesh, i.e. converts the template based buffers
	 * 			to OpenGL compatible buffers
	 */
	virtual void finalize();


private:

	/// The faces in the half edge mesh
	vector<HalfEdgeFace<VertexT, NormalT>*>    m_faces;

	/// The vertices of the mesh
	vector<HalfEdgeVertex<VertexT, NormalT>*>  m_vertices;

	/// The indexed of the newest inserted vertex
	int 					 m_globalIndex;

	void printStats();

//	void check_next_neighbor(HalfEdgeFace* f0,
//							 HalfEdgeFace* face,
//							 HalfEdge* edge,
//							 HalfEdgePolygon*);
//
//	void check_next_neighbor(HalfEdgeFace* f0,
//			                 HalfEdgeFace* face,
//			                 HalfEdge* edge,
//			                 vector<HalfEdgeFace*>& faces);

//	void extract_borders();
//	void generate_polygons();

//	void getArea(set<HalfEdgeFace*> &faces, HalfEdgeFace* face, int depth, int max);
//	void shiftIntoPlane(HalfEdgeFace* f);

//	bool check_face(HalfEdgeFace* f0, HalfEdgeFace* current);
//	bool isFlatFace(HalfEdgeFace* face);

//	int classifyFace(HalfEdgeFace* f);
//
//	void create_polygon(vector<int> &polygon,
//						hash_map< unsigned int, HalfEdge* >* edges);

//	void cluster(vector<planarCluster> &planes);
//	void optimizeClusters(vector<planarCluster> &clusters);
//
//	void classifyCluster(vector<planarCluster> &panes, list<list<planarCluster> > &objectCandidates);
//	void findNextClusterInRange(int s, vector<planarCluster> &clusters,
//	        planarCluster &start,
//	        list<planarCluster> &clustercluster,
//	        vector<bool> &markers);
//
//	virtual void finalize(vector<planarCluster> &planes);
//	virtual void finalize(list<list<planarCluster> > &objects);

private:

	/**
	 * @brief   Returns an edge that point to the edge defined
	 *          by the given vertices.
	 *
	 * @param v     The start vertex of an edge
	 * @param next  The end vertex of an edge
	 * @return      A pointer to an existing edge, or null if no suitable
	 *              edge was found.
	 */
	HEdge* halfEdgeToVertex(HVertex* v, HVertex* next);

//	int biggest_size;
//	HalfEdgePolygon* biggest_polygon;
//
//	float  current_d;
//	Normal current_n;
//	Vertex current_v;

};

} // namespace lssr


#include "HalfEdgeMesh.tcc"

#endif /* HALFEDGEMESH_H_ */
