/*
 * TSDFGrid.tcc
 *
 *  November 29, 2014
 *  Author: Tristan Igelbrink
 */

#include "TSDFGrid.hpp"

namespace lvr
{

template<typename VertexT, typename BoxT, typename TsdfT>
TsdfGrid(float cellSize,  BoundingBox<VertexT> bb, TsdfT* tsdf, size_t size, bool isVoxelsize = true);
	: HashGrid<VertexT, BoxT>(cellSize, bb, isVoxelsize)
{
	// Iterator over all points, calc lattice indices and add lattice points to the grid
	for(size_t i = 0; i < size; i++)
	{
		this->addLatticePoint(tsdf[size].x , tsdf[size].y, tsdf[size].z, tsdf[size].w);
	}

}

template<typename VertexT, typename BoxT, typename TsdfT>
TsdfGrid<VertexT, BoxT>::~TsdfGrid()
{

}

} /* namespace lvr */
