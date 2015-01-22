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
	this->setCoordinateScaling(1.0, 1.0, 1.0);
	cout << timestamp << "Started creating grid " << "Values: " << size << endl;
	//fill queryPoint vector and hash map
	for(size_t i = 0; i < size; i++)
	{
		VertexT position(tsdf[i].x, tsdf[i].y, tsdf[i].z);
		this->m_queryPoints.push_back(QueryPoint<VertexT>(position, tsdf[i].w));
		size_t hash_value = this->hashValue(tsdf[i].x, tsdf[i].y, tsdf[i].z);
		this->m_qpIndices[hash_value] = i;
		this->m_globalIndex++;
	}
	cout << timestamp << "Finished inserting" << endl;
	// Iterator over all points, calc lattice indices and add lattice points to the grid
	for(size_t i = 0; i < size; i++)
	{
		//#pragma omp task
		this->addLatticePoint(tsdf[i].x , tsdf[i].y, tsdf[i].z, tsdf[i].w);
	}
	cout << timestamp << "Finished creating grid" << endl;
}

template<typename VertexT, typename BoxT, typename TsdfT>
TsdfGrid<VertexT, BoxT, TsdfT>::~TsdfGrid()
{

}

} /* namespace lvr */
