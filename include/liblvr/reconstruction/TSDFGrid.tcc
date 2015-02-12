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
	cout << timestamp << "Started creating grid " << "Values: " << size << endl;
	size_t center_of_bb_x = (this->m_boundingBox.getXSize()/2) / this->m_voxelsize;
	size_t center_of_bb_y = (this->m_boundingBox.getYSize()/2) / this->m_voxelsize;
	size_t center_of_bb_z = (this->m_boundingBox.getZSize()/2) / this->m_voxelsize;
	//#pragma omp parallllell for
	for(size_t i = 0; i < size; i++)
	{
		// shift tsdf into global grid
		int global_x = tsdf[i].x + center_of_bb_x;
		int global_y = tsdf[i].y + center_of_bb_y;
		int global_z = tsdf[i].z + center_of_bb_z;
		VertexT position(global_x, global_y, global_z);
		this->m_queryPoints[i] = QueryPoint<VertexT>(position, tsdf[i].w);
		size_t hash_value = this->hashValue(global_x, global_y, global_z);
		this->m_qpIndices[hash_value] = i;
		this->m_globalIndex++;
	}
	cout << timestamp << "Finished inserting" << endl;
	// Iterator over all points, calc lattice indices and add lattice points to the grid
	for(size_t i = 0; i < size; i++)
	{
		int global_x = tsdf[i].x + center_of_bb_x;
		int global_y = tsdf[i].y + center_of_bb_y;
		int global_z = tsdf[i].z + center_of_bb_z;
		//#pragma omp task
		this->addLatticePoint(global_x , global_y, global_z, tsdf[i].w);
	}
	cout << timestamp << "Finished creating grid" << endl;
}

template<typename VertexT, typename BoxT, typename TsdfT>
TsdfGrid<VertexT, BoxT, TsdfT>::~TsdfGrid()
{

}

} /* namespace lvr */
