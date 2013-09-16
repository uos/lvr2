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
 * SlicerMesh.hpp
 *
 *  @date   21.08.2013
 *  @author Ann-Katrin Häuser (ahaeuser@uos.de)
 *  @author Henning Deeken (hdeeken@uos.de)
 *  @author Thomas Wiemann (twiemann@uos.de)
 */

#ifndef SLICER_H_
#define SLICER_H_

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
#include <CGAL/exceptions.h>

#include "geometry/Vertex.hpp"
#include "geometry/VertexTraits.hpp"

#include "io/Timestamp.hpp"
#include "io/Progress.hpp"
#include "io/Model.hpp"

using namespace std;

namespace lssr
{

/**
 * @brief Mesh Slicer
 **/
 
class MeshSlicer
{
	
public:
	
	typedef CGAL::Simple_cartesian<double> K;
	typedef K::FT FT;
	typedef K::Vector_3 Vector;
	typedef K::Plane_3 Plane;
	typedef K::Ray_3 Ray;
	typedef K::Line_3 Line;
	typedef K::Point_3 Point;
	typedef K::Triangle_3 Triangle;
	typedef K::Segment_3 Segment; 
	typedef std::list<Triangle>::iterator Iterator;
	typedef CGAL::AABB_triangle_primitive<K,Iterator> Primitive;
	typedef CGAL::AABB_traits<K, Primitive> AABB_triangle_traits;
	typedef CGAL::AABB_tree<AABB_triangle_traits> Tree;

	/**
	 * @brief   Creates an empty FusionMesh 
	 */
	MeshSlicer();

	/**
	 * @brief   Destructor.
	 */
	virtual ~MeshSlicer()
	{};

	/// Input Methods

	/**
     * @brief   Insert an entire mesh into the local fusion buffer. It is advised to call integrate() afterwards.
     *
     * @param   mesh      A pointer to the mesh to be inserted
     */
	virtual void addMesh(MeshBufferPtr model);
	
	/// AABB Tree & Slice Computation
	
	/**
     * @brief   build CGAL-AABB-Tree from global mesh
	 *
     */
	virtual void buildTree();

	/**
     * @brief   Calculates the query plane
     *     
     */
	virtual Plane getQueryPlane();
	
	/**
     * @brief   Calculate all intersection with the desired plane 
     *     
     * @param   segments 		A buffer for all intersection segments
     */
	virtual void computeIntersections(vector<Segment>& segments);
	
	/**
     * @brief   Calculate all traingles within the desired bounding box and projects them onto a plane
     *     
     * @param   segments 		A buffer for all intersection segments
     */
	virtual void computeProjections(vector<Segment>& segments);
	
	/**
     * @brief   Integrate the local buffer into the global fused mesh
	 *
     */
	virtual vector<float> compute2dSlice();

	/**
	* @brief   Integrate the local buffer into the global fused mesh
	 *
	*/
	virtual vector<float> compute2dProjection();

	/// Convenience Function
	
	/**
	* @brief   Insert an entire mesh into the local fusion buffer and integrate it imediately.
	*
	* @param   mesh      A pointer to the mesh to be inserted
	*/
	virtual vector<float> addMeshAndCompute2dSlice(MeshBufferPtr model);

	/**
     * @brief   Insert an entire mesh into the local fusion buffer and integrate it imediately.
     *
     * @param   mesh      A pointer to the mesh to be inserted
     */
	virtual vector<float> addMeshAndCompute2dProjection(MeshBufferPtr model);

	/// Parameter Methods

	/**
	 * Sets the dimension
	 *
	 * @param dim 	dimension to slice 
	 *
	 */
	 
	void setDimension(string dim) 
	{ 
		dimension = dim;
	};
	
	/**
	 * Sets the plane position 
	 *
	 * @param val 	value
	 *
	 */
	 	 
	void setValue(double val)
	{ 
		value = val;
		
	};
	
	/**
     * @brief   Reset the Input Data Storage.
     */
	virtual void clear();

private:

	/// Input Data
	 size_t num_verts, num_faces;
	 vector<float> vertices;
	 vector<unsigned int>  faces;
	
	/// Intersection Buffer 
	vector<Segment> segments;
	vector<float> output;

	/// CGAL AABB Tree
	Tree	tree;

	/// Parameter
	string	dimension;
	double	coord_x;
	double	coord_y;
	double	coord_z;
	double	value;
};

} // namespace lssr

#include "MeshSlicer.cpp"

#endif /* MESHSLICER_H_ */
