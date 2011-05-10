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

template<typename VertexT, typename NormalT>
class HalfEdgeFace
{
public:
	HalfEdgeFace() {};
	HalfEdgeFace(const HalfEdgeFace<VertexT, NormalT> &o);

	void calc_normal();
	void interpolate_normal();

	void getVertexNormals(vector<NormalT> &n);
	void getVertices(vector<VertexT> &v);
	void getAdjacentFaces(vector<HalfEdgeFace<VertexT, NormalT>* > &adj);

	NormalT getFaceNormal();
	NormalT getInterpolatedNormal();

	VertexT getCentroid();

	float getArea();

	HalfEdge<HalfEdgeVertex<VertexT, NormalT>, HalfEdgeFace<VertexT, NormalT> >* m_edge;

	bool 							m_used;
	vector<int> 					m_indices;
	int 							m_index[3];
	int 							m_mcIndex;
	int 							m_texture_index;

	//unsigned int face_index;

	size_t							m_face_index;

	NormalT							m_normal;
};

}

#include "HalfEdgeFace.tcc"

#endif /* HALFEDGEFACE_H_ */
