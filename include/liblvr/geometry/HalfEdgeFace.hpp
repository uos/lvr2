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
 * HalfEdgeFace.hpp
 *
 *  @date 03.12.2008
 *  @author Kim Rinnewitz (krinnewitz@uos.de)
 *  @author Sven Schalk (sschalk@uos.de)
 *  @author Thomas Wiemann (twiemann@uos.de)
 */

#ifndef HALFEDGEFACE_H_
#define HALFEDGEFACE_H_

#include <vector>
#include <set>

using namespace std;

#include "HalfEdgeVertex.hpp"
#include "Normal.hpp"
#include "HalfEdge.hpp"
#include "Region.hpp"


namespace lvr
{

/**
 * @brief A face in a half edge mesh.
 *
 * Besides the classical linking, this class stores
 * some additional information that becomes handy when
 * implementing optimization algorithms
 */
template<typename VertexT, typename NormalT>
class HalfEdgeFace
{
public:

    typedef HalfEdge<HalfEdgeVertex<VertexT, NormalT>, HalfEdgeFace<VertexT, NormalT> >* EdgePtr;
    typedef HalfEdgeVertex<VertexT, NormalT>* VertexPtr;
    typedef HalfEdgeFace<VertexT, NormalT>*  FacePtr;
    typedef vector<HalfEdgeFace<VertexT, NormalT>* >FaceVector;
     /**
     * @brief   Constructs an empty face
     */
	HalfEdgeFace() :
	    m_invalid(false), m_edge(0),
	    m_used(false), m_region(-1),
	    m_texture_index(-1),
	    m_face_index(-1) {};

	/**
	 * @brief Destructor
	 */
	virtual ~HalfEdgeFace();

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
	void getAdjacentFaces(FaceVector &adj);

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
	 * @brief	Indexed edge access (reading)
	 */
	virtual HalfEdge<HalfEdgeVertex<VertexT, NormalT>, HalfEdgeFace<VertexT, NormalT> >* operator[](const int &index) const;

	/**
	 * @brief	Indexed vertex access (reading)
	 */
	virtual HalfEdgeVertex<VertexT, NormalT>* operator()(const int &index) const;

	/**
	 * @brief Returns the size of the face
	 */
	float getArea();

	/**
	 * @brief Returns the "d" from the pane equation "ax + by + cx + d = 0"
	 */
	float getD();

	/**
	 * @brief Returns true, if one of the face's edges has no adjacent face
	 */
	bool isBorderFace();

	/**
	 * @brief Returns the id of this face in MeshBuffer after finalize step.
	 */
	int getBufferID();

	/**
	 * @brief Sets the id of this face in MeshBuffer after finalize step.
	 * @param id The id of this face in MeshBuffer after finalize step
	 */
	void setBufferID(unsigned int id);

	/// A pointer to a surrounding half edge
	EdgePtr m_edge;

	/// A vector containing the indices of face vertices (currently redundant)
	//vector<int> 					m_indices;

	/// A three pointers to the face vertices
	//int 							m_index[3];

	/// The index of the face's texture
	int 							m_texture_index;

	/// The region of the face
	long int		                m_region;

	/// used for region growing
	bool							m_used;

	/// The number of the face in the half edge mesh (convenience only, will be removed soon)
	size_t							m_face_index;

	/// The face normal
	NormalT							m_normal;

	bool                            m_invalid;

private:

	unsigned int                    buffer_id;
};

}

#include "HalfEdgeFace.tcc"

#endif /* HALFEDGEFACE_H_ */
