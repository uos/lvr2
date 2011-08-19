/*
 * Region.h
 *
 *  Created on: 18.08.2011
 *      Author: PG2011
 */

#ifndef REGION_H_
#define REGION_H_

namespace lssr {

/**
 * @brief 	This class represents a region.
 */
template<typename VertexT, typename NormalT>
class Region
{

public:

	/**
	 * @brief	constructor.
	 *
	 * @param 	faces	a list of all faces of the region
	 */
	virtual Region(vector<HalfEdgeFace<VertexT, NormalT>*>    faces);

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

	/// Indicates if the region was dragged into a regression plane
	bool m_inPlane;

private:
	/// The faces in the region
	vector<HalfEdgeFace<VertexT, NormalT>*>    m_faces;

};
}

#include "Region.tcc"

#endif /* REGION_H_ */
