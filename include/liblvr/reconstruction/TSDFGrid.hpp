/*
 * TSDFGrid.hpp
 *
 *  November 29, 2014
 *  Author: Tristan Igelbrink
 */

#ifndef _TSDFGRID_HPP_
#define _TSDFGRID_HPP_

#include "HashGrid.hpp"

namespace lvr
{

template<typename VertexT, typename BoxT, typename TsdfT>
class TsdfGrid: public HashGrid<VertexT, BoxT>
{
public:

	typedef unordered_map<size_t, BoxT*> box_map;
	
	/// Typedef to alias iterators for box maps
	typedef typename unordered_map<size_t, BoxT*>::iterator  box_map_it;
	
	TsdfGrid(float cellSize,  BoundingBox<VertexT> bb, TsdfT* tsdf, size_t size, bool isVoxelsize = true);
	virtual ~TsdfGrid();
	void addTSDFLatticePoint(int index_x, int index_y, int index_z, float distance);
	inline int calcIndex(float f)
    {
        return f < 0 ? f-.5:f+.5;
    }

};

} /* namespace lvr */

#include "TSDFGrid.tcc"


#endif /* INCLUDE_LIBLVR_RECONSTRUCTION_TsdfGrid_HPP_ */
