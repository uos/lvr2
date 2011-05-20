/*
 * HalfEdgeFace.h
 *
 *  Created on: 03.12.2008
 *      Author: Thomas Wiemann
 */

#ifndef HALFEDGEFACE_H_
#define HALFEDGEFACE_H_

#include <vector>
#include <set>

using namespace std;

#include "HalfEdgeVertex.hpp"
#include "Normal.hpp"
#include "HalfEdge.hpp"

namespace lssr
{

template<typename VertexT, typename NormalT>
class HalfEdgeVertex;

/**
 * @brief A face in a half edge mesh.
 *
 * Besides the classical linking, this class stores
 * some additional information that becomes handy when
 * implementin optimization algorithms
 */
template<typename VertexT, typename NormalT>
class HalfEdgeFace
{
public:

    /**
     * @brief   Constructs an empty face
     */
	HalfEdgeFace() {};

	/**
	 * @brief   Copy constructor
	 *
	 * @param o The mesh to copy
	 */
	HalfEdgeFace(const HalfEdgeFace<VertexT, NormalT> &o);

	/**
	 * @brief   Calculates a face normal
	 */
	void calc_normal();

	/**
	 * @brief   Calculates an interpolated normal (mean of three face normals)
	 */
	void interpolate_normal();

	/**
	 * @brief   Delivers the three vertex normals of the face
	 * @param n     A vector to store the vertex normals
	 */
	void getVertexNormals(vector<NormalT> &n);

	/**
	 * @brief   Delivers the three vertices of the face.
	 *
	 * @param v     A vector to store the vertices
	 */
	void getVertices(vector<VertexT> &v);

	/**
	 * @brief   Returns the adjacent face of this face
	 *
	 * @param adj   A vector to store the adjacent vertices
	 */
	void getAdjacentFaces(vector<HalfEdgeFace<VertexT, NormalT>* > &adj);

	/**
	 * @brief  Returns the face normal
	 */
	NormalT getFaceNormal();

	/**
	 * @brief Returns an interpolated normal (mean of the three vertex normals)
	 */
	NormalT getInterpolatedNormal();

	/**
	 * @brief Returns the centroid of the face
	 */
	VertexT getCentroid();

	/**
	 * @brief Returns the size of the face
	 */
	float getArea();

	/// A pointer to a surrounding half edge
	HalfEdge<HalfEdgeVertex<VertexT, NormalT>, HalfEdgeFace<VertexT, NormalT> >* m_edge;

	/// Used for clustering. True, if the face was already visited
	bool 							m_used;

	/// A vector containing the indices of face vertices (currently redundant)
	vector<int> 					m_indices;

	/// A three pointers to the face vertices
	int 							m_index[3];

	/// The index of the face's texture
	int 							m_texture_index;

	/// The number of the face in the half edge mesh (convenience only, will be removed soon)
	size_t							m_face_index;

	/// The face normal
	NormalT							m_normal;
};

}

#include "HalfEdgeFace.tcc"

#endif /* HALFEDGEFACE_H_ */
