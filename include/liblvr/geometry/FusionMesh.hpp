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
 * FusionMesh.hpp
 *
 *  @date   11.07.2013
 *  @author Ann-Katrin Häuser (ahaeuser@uos.de)
 *  @author Henning Deeken (hdeeken@uos.de)
 *  @author Thomas Wiemann (twiemann@uos.de)
 */

#ifndef FUSIONMESH_H_
#define FUSIONMESH_H_

#include <boost/unordered_map.hpp>

#include <vector>
#include <stack>
#include <set>
#include <list>
#include <map>
#include <unordered_map>
#include <iostream>
#include <sstream>
#include <float.h>
#include <math.h>
#include <algorithm>
#include <queue>

#include <glu.h>
#include <glut.h>

#include <CGAL/Simple_cartesian.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_triangle_primitive.h>

#include "Vertex.hpp"
#include "VertexTraits.hpp"
#include "Normal.hpp"
#include "BaseMesh.hpp"

#include "FusionVertex.hpp"
#include "FusionFace.hpp"
//#include "FusionEdge.hpp"

#include "io/Timestamp.hpp"
#include "io/Progress.hpp"
#include "io/Model.hpp"

using namespace std;

namespace lvr
{

template<typename VertexT, typename NormalT> class FusionVertex;
template<typename VertexT, typename NormalT> class FusionFace;

/**
 * @brief Implementation of a mesh structure that can be used to incrementally fuse different meshes into one.
 */
 
template<typename VertexT, typename NormalT> class FusionMesh : public BaseMesh<VertexT, NormalT>
{
	
public:
	
	typedef FusionFace<VertexT, NormalT> FFace;
	typedef FusionVertex<VertexT, NormalT> FVertex;
	
	typedef CGAL::Simple_cartesian<double> K;
	typedef K::FT FT;
	typedef K::Ray_3 Ray;
	typedef K::Line_3 Line;
	typedef K::Point_3 Point;
	typedef K::Triangle_3 Triangle;
	typedef std::list<Triangle>::iterator Iterator;
	typedef CGAL::AABB_triangle_primitive<K,Iterator> Primitive;
	typedef CGAL::AABB_traits<K, Primitive> AABB_triangle_traits;
	typedef CGAL::AABB_tree<AABB_triangle_traits> Tree;

	struct cmpVertices {
		bool operator()(const VertexT& a, const VertexT& b) const 
		{
		/*	
			   return ( a.x < b.x ) 
                || ( ( this->x - other.x <= 0.00001 )
                    && ( ( this->y < other.y ) 
                        || ( ( this->z < other.z )
                            && ( this->y - other.y <= 0.00001 ) ) ) );
			*/
			
			/*
			cout << "comparing " << a.x << " == "<<b.x << " and " << a.y << " == " <<  b.y << "and " << a.z << " == " << b.z << endl; 
			if(((a.x == b.x) && (a.y == b.y) && (a.z == b.z)))
			{
				cout << "EQUAL" << endl << endl;
			}
			else
			{
				cout << "unequal" << endl;
			}
			*/
			return !((a.x == b.x) && (a.y == b.y) && (a.z == b.z));
		}
	};
	
	struct hashVertices {
		std::size_t operator()(VertexT a) const {
     
			size_t hash = a.x + a.y + a.z; //std::hash<double>((double) a.x + a.y + a.z);
			return hash;
		}	
	};

	typedef map<VertexT, size_t> Map;
	typedef typename map<VertexT, size_t>::iterator MapIterator;
	
	//typedef map<VertexT, size_t, cmpVertices> Map;
	//typedef typename map<VertexT, size_t, cmpVertices>::iterator MapIterator;

	//typedef unordered_map<VertexT, size_t, hashVertices> Map;
	//typedef typename unordered_map<VertexT, size_t, hashVertices>::iterator MapIterator;



	/**
	 * @brief   Creates an empty FusionMesh 
	 */
	FusionMesh();

	/**
	 * @brief   Creates a FusionMesh from the given mesh buffer
	 */
	FusionMesh(MeshBufferPtr model);

	/**
	 * @brief   Destructor.
	 */
	virtual ~FusionMesh() {};

	// Mesh Construction Methods

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
     * @brief   Insert a new triangle into the mesh
     *
     * @param   a       The first vertex of the triangle
     * @param   b       The second vertex of the triangle
     * @param   c       The third vertex of the triangle
     * @param   f       A pointer to the created face
     */
	virtual void addTriangle(uint a, uint b, uint c, FFace* &f);
	
	/**
     * @brief   Insert an entire mesh into the local fusion buffer. It is advised to call integrate() afterwards.
     *
     * @param   mesh      A pointer to the mesh to be inserted
     */
	virtual void addMesh(MeshBufferPtr model);
	
// Integration Methods
	
	/**
	 * @brief 	This method should be called every time
	 * 			a vertex is transferred from the local buffer into the global buffer
	 *
	 * @param	v 		A FusionVertex from the local buffer.
	 */
	virtual void addGlobalVertex(FVertex *v);
	
	/**
     * @brief   Insert a new triangle into the mesh and change vertex indeces by increment
     *
     * @param   f       A face from the local buffer
     * @param   i		 The increment that has to be used to shift the face indices properly. Ususally i = current global vertex buffer size.
     */
	virtual void addGlobalTriangle(FFace *f, int i);	
	
	/**
     * @brief   build CGAL-AABB-Tree from global mesh
	 *
     */
	virtual void buildTree();
	
	/**
     * @brief   build map of global vertices with global buffer index
	 *
     */
	virtual void buildVertexMap();
	
	/**
     * @brief   Insert a new triangle into the the tree
     *
     * @param   remote_faces 			A buffer for all faces that can be added to the global buffer directly
     * @param   integration_faces 		A buffer for all faces that need to be integrated 
     */
	virtual void sortFaces(vector<FFace*>& remote_faces, vector<FFace*>& integration_faces );
	
	/**
     * @brief   Integrate the local buffer into the global fused mesh
	 *
     */
	virtual void integrate();
	
	/**
     * @brief   Integrate the local buffer into the global mesh, by simply adding all vertices and shifting the face indices
	 *
     */
	virtual void lazyIntegrate();
	
	/**
     * @brief   Integrate remote faces
	 *
     */
	virtual void remoteIntegrate(vector<FFace*>& faces);
	
	/**
     * @brief   Insert an entire mesh into the local fusion buffer and integrate it imediately.
     *
     * @param   mesh      A pointer to the mesh to be inserted
     */
	virtual void addMeshAndIntegrate(MeshBufferPtr model);
	
	/**
     * @brief   Insert an entire mesh into the local fusion buffer and lazyintegrate it imediately.
     *
     * @param   mesh      A pointer to the mesh to be inserted
     */
	virtual void addMeshAndLazyIntegrate(MeshBufferPtr model);
	
	/**
	 * @brief 	Finalizes a mesh, i.e. converts the template based buffers
	 * 			to OpenGL compatible buffers
	 */
	virtual void finalize();


// Parameter Methods

	/**
	 * Sets the distance treshold used for AABB search
	 *
	 * @param t 	distance treshold
	 *
	 */
	 
	void setDistanceThreshold(double_t t) {threshold = t*t;};

private:

	/// The faces in the fusion buffer 
	vector<FusionFace<VertexT, NormalT>*>   m_local_faces;

	/// The vertices of the fusion buffer
	vector<FusionVertex<VertexT, NormalT>*>   m_local_vertices;

	/// The length of the local vertex buffer
	size_t                                      m_local_index;

	/// The faces in the fused mesh
	vector<FusionFace<VertexT, NormalT>*>     m_global_faces;

	/// The vertices of the fused mesh
	vector<FusionVertex<VertexT, NormalT>*>   m_global_vertices;
	///  The length of the global vertex buffer
	size_t                                      m_global_index;

	/// FaceBuffer used during integration process
	vector<FFace*> remote_faces; 
    vector<FFace*> integration_faces; 
    int redundant_faces;
	int special_case_faces;	

	/// The CGAL AABB Tree
	Tree		tree;
	/// The Map with all global vertices
	Map			global_vertices_map;
	
	/// Squared maximal distance for fusion
	double_t	threshold;

	/**
     * @brief   Reset the the local buffer e.g. after integration or at initialization.
     */
	virtual void clearLocalBuffer();

	/**
     * @brief   Reset the the global buffer e.g. at initialization.
     */
	virtual void clearGlobalBuffer();
	
	/**
     * @brief   Prints the current status of the local buffer on the console.
     */
	virtual void printLocalBufferStatus();
	
	/**
     * @brief   Prints the current status of the local buffer on the console.
     */
	virtual void printGlobalBufferStatus();
	
	/**
     * @brief   Prints the current status of the face sorting process on the console.
     */
	virtual void printFaceSortingStatus();

// unused methods, yet necessary due to BaseMesh interface

	/**
	 * @brief	Flip the edge between vertex index v1 and v2
	 *
	 * @param	v1	The index of the first vertex
	 * @param	v2	The index of the second vertex
	 */
	virtual void flipEdge(uint v1, uint v2);
};

} // namespace lvr


#include "FusionMesh.cpp"

#endif /* FUSIONMESH_H_ */
