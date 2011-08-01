/*
 * HalfEdgeMesh.h
 *
 *  Created on: 13.11.2008
 *      Author: Thomas Wiemann
 */

#ifndef HALFEDGEMESH_H_
#define HALFEDGEMESH_H_

#include <boost/unordered_map.hpp>

#include <vector>
#include <stack>
#include <set>
#include <list>
#include <sstream>
#include <float.h>
#include <math.h>

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
	 * @brief	This method should be called every time
	 * 			a vertex is deleted
	 *
	 * @param	v	The vertex to delete.
	 */
	virtual void deleteVertex(HVertex* v);

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
	 *  @brief	Adds a new face
	 *
	 *  @param	v1	First vertex of the new face
	 *  @param	v2	Second vertex of the new face
	 *  @param	v3	Third vertex of the new face
	 */
	virtual void addFace(HVertex* v1, HVertex* v2, HVertex* v3);

	/**
	 * @brief	Delete the given edge
	 *
	 * @param	edge		The edge to delete
	 * @param	deletePair	Whether to delete edge->pair or not
	 */
	virtual void deleteEdge(HEdge* edge, bool deletePair = true);

	/**
	 * @brief	Delete a face from the mesh
	 * 			Also deletes dangling vertices and Edges
	 *
	 * @param	f		The face to be deleted
	 */
	virtual void deleteFace(HFace* f);

	/**
	 * @brief	Collapse the given edge
	 *
	 * @param	edge	The edge to collapse
	 */
	virtual void collapseEdge(HEdge* edge);

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
	 * @brief	Starts a region growing and returns the number of connected faces
	 * 			Faces are connected means they share a common edge - a point is not
	 *			a connection in this context
	 *
	 * @param	start_face	The face from which the region growing is started
	 *
	 * @param	region		The region number to apply to the faces of the found region
	 *
	 * @return	Returns the size of the region - 1 (the start face is not included)
	 */
	virtual int regionGrowing(HFace* start_face, int region);

	/**
	 * @brief	Starts a region growing wrt the angle between the faces and returns the
	 * 			number of connected faces. Faces are connected means they share a common
	 * 			edge - a point is not a connection in this context
	 *
	 * @param	start_face	The face from which the region growing is started
	 *
	 * @param	normal		The normal to refer to
	 *
	 * @param	angle		the maximum angle allowed between two faces
	 *
	 * @param	region		The region number to apply to the faces of the found region
	 *
	 * @return	Returns the size of the region - 1 (the start face is not included)
	 */
	virtual int regionGrowing(HFace* start_face, NormalT &normal, float &angle, int region);

	/**
	 * @brief	Applies region growing and regression plane algorithms and deletes small
	 * 			regions
	 *
	 * @param iterations	The number of iterations to use
	 */
	virtual void optimizePlanes(int iterations);

	/**
	 * @brief	Calculates a regression plane for the given region and projects all
	 * 			vertices of the region into this plane.
	 *
	 * @param	region	The region to improve
	 */
	virtual void regressionPlane(int region);

	/**
	 * @brief	Deletes all faces belonging to the given region
	 *
	 * @param	region	The region to delete
	 */
	virtual void deleteRegion(int region);

	/**
	 * @brief	Removes artifacts in the mesh that are not connected to the main mesh
	 *
	 * @param	threshold	Specifies the maximum number of faces
	 * 						which will be detected as an artifact
	 */
	virtual void removeDanglingArtifacts(int threshold);

	/**
	 * @brief	Finds the contour of a hole starting from a given starting point
	 *
	 * @param start		The starting point
	 * @param end		The end point
	 * @param contour	A sequence of vertices defining the contour of the hole
	 *
	 * @return	The length of the contour (number of edges)
	 */
	virtual vector<HVertex*>  simpleDetectHole(HEdge* start);

	/**
	 * @brief 	Fills a hole
	 *
	 * @param	contour	The contour of the hole to fill
	 *
	 */
	virtual void fillHole(vector<HVertex*> contour);


    /**
     * @brief   Takes a list of vertices as the border of a polygon
     *          and returns a triangle tesselation
     */
    void tesselate(void);


	/**
	 * @brief 	Finalizes a mesh, i.e. converts the template based buffers
	 * 			to OpenGL compatible buffers
	 */
	virtual void finalize();


private:

	/// The faces in the half edge mesh
	vector<HalfEdgeFace<VertexT, NormalT>*>    m_faces;

	/// The vertices of the mesh
	//vector<HalfEdgeVertex<VertexT, NormalT>*>  m_vertices;
	vector<HalfEdgeVertex<VertexT, NormalT>*> m_vertices;

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
