/*
 * PointsetGrid.hpp
 *
 *  Created on: Nov 27, 2014
 *      Author: twiemann
 */

#ifndef _POINTSETGRID_HPP_
#define _POINTSETGRID_HPP_

#include "HashGrid.hpp"

#include "reconstruction/PointsetSurface.hpp"

namespace lvr
{

template<typename VertexT, typename BoxT>
class PointsetGrid: public HashGrid<VertexT, BoxT>
{
public:
	PointsetGrid(float cellSize, BoundingBox<VertexT> boundingBox, typename PointsetSurface<VertexT>::Ptr surface, bool isVoxelsize = true);
	virtual ~PointsetGrid();

	void calcDistanceValues();

private:
	typename PointsetSurface<VertexT>::Ptr		m_surface;
};

} /* namespace lvr */

#endif /* INCLUDE_LIBLVR_RECONSTRUCTION_POINTSETGRID_HPP_ */
