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
TsdfGrid<VertexT, BoxT, TsdfT>::TsdfGrid(float cellSize,  BoundingBox<VertexT> bb, TsdfT* tsdf, size_t size, bool isVoxelsize)
	: HashGrid<VertexT, BoxT>(cellSize, bb, isVoxelsize)
{
	this->setCoordinateScaling(1.0, -1.0, 1.0);
	cout << timestamp << "Started creating grid " << "Values: " << size << endl;
	// Iterator over all points, calc lattice indices and add lattice points to the grid
	for(size_t i = 0; i < size; i++)
	{
		#pragma omp task
		this->addLatticePoint(tsdf[i].x , tsdf[i].y, tsdf[i].z, tsdf[i].w);
	}
	cout << timestamp << "Finished creating grid" << endl;
}

template<typename VertexT, typename BoxT, typename TsdfT>
TsdfGrid<VertexT, BoxT, TsdfT>::~TsdfGrid()
{

}

} /* namespace lvr */
