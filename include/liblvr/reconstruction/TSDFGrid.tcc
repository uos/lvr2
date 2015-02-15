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
	this->m_queryPoints.resize(size);
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
		addTSDFLatticePoint(global_x , global_y, global_z, tsdf[i].w);
	}
	cout << timestamp << "Finished creating grid" << endl;
}

template<typename VertexT, typename BoxT, typename TsdfT>
void TsdfGrid<VertexT, BoxT, TsdfT>::addTSDFLatticePoint(int index_x, int index_y, int index_z, float distance)
{
	size_t hash_value;

	unsigned int INVALID = BoxT::INVALID_INDEX;

	float vsh = 0.5 * this->m_voxelsize;
	//float vsh = m_voxelsize;

	// Some iterators for hash map accesses
	box_map_it it;
	box_map_it neighbor_it;

	// Values for current and global indices. Current refers to a
	// already present query point, global index is id that the next
	// created query point will get
	unsigned int current_index = 0;

	int dx, dy, dz;

	// Get min and max vertex of the point clouds bounding box
	VertexT v_min = this->m_boundingBox.getMin();
	VertexT v_max = this->m_boundingBox.getMax();

	/*int e;
	m_extrude ? e = 8 : e = 1;
	for(int j = 0; j < e; j++)
	{*/
		// Get the grid offsets for the neighboring grid position
		// for the given box corner
		

		hash_value = this->hashValue(index_x, index_y, index_z);

		//it = m_cells.find(hash_value);
		//if(it == m_cells.end())
		//{
			//Calculate box center .. useless
			VertexT box_center(
					(index_x * this->m_voxelsize + v_min[0] * vsh),
					(index_y * this->m_voxelsize + v_min[1] * vsh),
					(index_z * this->m_voxelsize + v_min[2] * vsh));

			//Create new box
			BoxT* box = new BoxT(box_center);

			//Setup the box itself
			for(int k = 0; k < 8; k++)
			{
				//Find point in Grid
				//current_index = findQueryPoint(k, index_x + dx, index_y + dy, index_z + dz);
				dx = TSDFCreateTable[k][0];
				dy = TSDFCreateTable[k][1];
				dz = TSDFCreateTable[k][2];
				size_t corner_hash = this->hashValue(index_x + dx, index_y + dy, index_z + dz);
				auto qp_index_it = this->m_qpIndices.find(corner_hash);
				//If point exist, save index in box
				if(qp_index_it != this->m_qpIndices.end()) 
					box->setVertex(k, qp_index_it->second);
				else
					return;
			}

			//Set pointers to the neighbors of the current box
			int neighbor_index = 0;
			size_t neighbor_hash = 0;

			for(int a = -1; a < 2; a++)
			{
				for(int b = -1; b < 2; b++)
				{
					for(int c = -1; c < 2; c++)
					{

						//Calculate hash value for current neighbor cell
						neighbor_hash = this->hashValue(index_x + a,
								index_y + b,
								index_z + c);

						//Try to find this cell in the grid
						neighbor_it = this->m_cells.find(neighbor_hash);

						//If it exists, save pointer in box
						if(neighbor_it != this->m_cells.end())
						{
							box->setNeighbor(neighbor_index, (*neighbor_it).second);
							(*neighbor_it).second->setNeighbor(26 - neighbor_index, box);
						}

						neighbor_index++;
					}
				}
			}

			this->m_cells[hash_value] = box;
		//}
	//}

}

template<typename VertexT, typename BoxT, typename TsdfT>
TsdfGrid<VertexT, BoxT, TsdfT>::~TsdfGrid()
{

}

} /* namespace lvr */
