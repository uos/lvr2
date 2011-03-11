/*
 * BaseMesh.h
 *
 *  Created on: 03.02.2011
 *      Author: Thomas Wiemann
 */

#ifndef BASEMESH_H_
#define BASEMESH_H_

namespace lssr {

/**
 * @brief 	Abstract interface class for dynamic triangle meshes.
 * 			The surface reconstruction algorithm can handle all
 *			all data structures that allow sequential insertion
 *			all of indexed triangles.
 */
template<typename VertexType, typename IndexType>
class BaseMesh
{
public:

	/**
	 * @brief 	This method should be called every time
	 * 			a new vertex is created.
	 *
	 * @param	v 		A supported vertex type. All used vertex types
	 * 					must support []-access.
	 */
	virtual void addVertex(VertexType v) = 0;

	/**
	 * @brief 	This method should be called every time
	 * 			a new vertex is created to ensure that vertex
	 * 			and normal buffer always have the same size
	 *
	 * @param	n 		A supported vertex type. All used vertex types
	 * 					must support []-access.
	 */
	virtual void addNormal(VertexType n) = 0;

	/**
	 * @brief 	Insert a new triangle into the mesh
	 *
	 * @param	a 		The first vertex of the triangle
	 * @param 	b		The second vertex of the triangle
	 * @param	c		The third vertex of the triangle
	 */
	virtual void addTriangle(IndexType a, IndexType b, IndexType c) = 0;
};
}

#endif /* BASEMESH_H_ */
