/*
 * Region.h
 *
 *  Created on: 18.08.2011
 *      Author: PG2011
 */

#ifndef REGION_H_
#define REGION_H_

#include "Vertex.hpp"
#include "Normal.hpp"
#include "HalfEdgeVertex.hpp"
#include "HalfEdgeFace.hpp"
#include <vector>
#include <stack>


namespace lssr {

/**
 * @brief 	This class represents a region.
 */
template<typename VertexT, typename NormalT>
class Region
{

public:

	typedef HalfEdgeFace<VertexT, NormalT> HFace;
	typedef HalfEdgeVertex<VertexT, NormalT> HVertex;
	typedef HalfEdge<HVertex, HFace> HEdge;

	/**
	 * @brief Adds a face to the region.
	 *
	 * @param	f	the face to add
	 */
	virtual void addFace(HFace* f);

	/**
	 * @brief Finds all contours of the region (outer contour + holes)
	 *
	 * @param	epsilon	controls the number of points used for a contour
	 *
	 * @return 	a list of all contours
	 */
	virtual vector<stack<HVertex*> > getContours(float epsilon);

	/**
	 * @brief calculates a valid normal of the region
	 *
	 * @return a normal of the region
	 */
	virtual NormalT getNormal();

	/*
	 * @brief caluclates a regression plane for the region and fits all faces into this plane
	 *
	 */
	virtual void regressionPlane();

	/**
	 * @brief destructor.
	 */
	virtual ~Region();

	/// Indicates if the region was dragged into a regression plane
	bool m_inPlane;

//private:
	/// The faces in the region
	vector<HFace*>    m_faces;

};
}

#include "Region.tcc"

#endif /* REGION_H_ */
