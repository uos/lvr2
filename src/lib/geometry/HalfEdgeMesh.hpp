/* Copyright (C) 2011 Uni Osnabrück
 * This file is part of the LAS VEGAS Reconstruction Toolkit,
 *
 * LAS VEGAS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * LAS VEGAS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA
 */


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

#include <glu.h>
#include <glut.h>

using namespace std;

#include "Vertex.hpp"
#include "Normal.hpp"
#include "BaseMesh.hpp"

#include "HalfEdgeVertex.hpp"
#include "HalfEdge.hpp"
#include "HalfEdgeFace.hpp"

//#include "HalfEdgePolygon.h"
#include "../io/Progress.hpp"

#include "Region.hpp"
#include "Tesselator.hpp"
#include "Texture.hpp"
#include "ColorVertex.hpp"

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
	 *
	 * @param	pm	a pointer to the point cloud manager
	 */
	HalfEdgeMesh(PointCloudManager<VertexT, NormalT>* pm);

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
	virtual int regionGrowing(HFace* start_face, Region<VertexT, NormalT>* region);

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
	virtual int regionGrowing(HFace* start_face, NormalT &normal, float &angle, Region<VertexT, NormalT>* region);

	/**
	 * @brief	Applies region growing and regression plane algorithms and deletes small
	 * 			regions
	 *
	 * @param iterations	The number of iterations to use
	 */
	virtual void optimizePlanes(int iterations, float normalThreshold, int minRegionSize = 50, int smallRegionSize = 0, bool remove_flickering = true);

	/**
	 * @brief	Deletes all faces belonging to the given region
	 *
	 * @param	region	The region to delete
	 */
	virtual void deleteRegion(Region<VertexT, NormalT>* region);


	/**
	 * @brief	Removes artifacts in the mesh that are not connected to the main mesh
	 *
	 * @param	threshold	Specifies the maximum number of faces
	 * 						which will be detected as an artifact
	 */
	virtual void removeDanglingArtifacts(int threshold);


	/**
	 *	@brief	drags the points of the given plane onto the given intersection if those points lay at
	 *			the border between the two given regions
	 *
	 *	@param	plane			the region which points are dragged
	 *	@param	neighbor_region	the region of the other plane belonging to the intersection line
	 *	@param	x				a point on the intersection line
	 *	@param	direction		the direction of the intersection line
	 */
	virtual void dragOntoIntersection(Region<VertexT, NormalT>* plane, Region<VertexT, NormalT>* neighbor_region, VertexT& x, VertexT& direction);

	/**
	 * @brief 	optimizes the plane intersections
	 */
	virtual void optimizePlaneIntersections();

	/**
	 * @brief	finds all contours in the mesh
	 *
	 * @param	epsilon	controls the number of points used for a contour
	 *
	 * @return 	a list of all contours
	 */
	virtual vector<vector<HVertex*> > findAllContours(float epsilon);

	/**
	 * @brief 	Finalizes a mesh, i.e. converts the template based buffers
	 * 			to OpenGL compatible buffers
	 */
	virtual void finalize();

	/**
	 * @brief 	Finalizes a mesh, i.e. converts the template based buffers
	 * 			to OpenGL compatible buffers. Furthermore all regions that
     * 			belong to a regression plane are retesselated to reduce triangles.
	 */
	virtual void finalizeAndRetesselate(bool genTextures);

	/**
	 * @brief 	fills all holes
	 *
	 * @param 	max_size 	the maximum size of a hole
	 */
	virtual void fillHoles( size_t max_size );

	/**
	 * @brief	Collapse the given edge safely
	 *
	 * @param	edge	The edge to collapse
	 *
	 * @return	true if the edge was collapsed, false otherwise
	 */
	virtual bool safeCollapseEdge(HEdge* edge);

	/**
	 * restores the position of every triangle in a plane if it has been modified
	 */
	virtual void restorePlanes();

	void enableRegionColoring() { m_colorRegions = true;}

    //virtual void regionsToBuffer(
    //            float **vertex, 
    //            float **normal,
    //            uchar **color,
    //            float **texture,
    //            unsigned int **index, 
    //            unsigned int **textureIndex, 
    //            size_t     &vncSize, 
    //            size_t   &indexSize, 
    //            vector<int> &regions);

    //virtual void retesselateRegionsToBuffer(size_t &vncBufferSize, 
    //                                size_t &indexBufferSize,
    //                                vector<int> plane_regions , bool genTextures=false);

	void tester();

private:

	/// The faces in the half edge mesh
	vector<HalfEdgeFace<VertexT, NormalT>*>     m_faces;

	/// The vertices of the mesh
	vector<HalfEdgeVertex<VertexT, NormalT>*>   m_vertices;

	/// The regions in the half edge mesh
	vector<Region<VertexT, NormalT>*>           m_regions;

	/// The indexed of the newest inserted vertex
	size_t                                      m_globalIndex;

	/// Indicates if regions will be colored in the ply format
	bool                                        m_colorRegions;

	///TODO: no longer used... ask fotte
    bool m_planesOptimized;

	/// a pointer to the point cloud manager
	PointCloudManager<VertexT, NormalT>* m_pointCloudManager;

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

};

} // namespace lssr


#include "HalfEdgeMesh.tcc"

#endif /* HALFEDGEMESH_H_ */
