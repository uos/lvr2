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
TsdfGrid<VertexT, BoxT, TsdfT>::TsdfGrid(float cellSize,  BoundingBox<VertexT> bb, TsdfT* tsdf, size_t size,
										int shiftX, int shiftY, int shiftZ, 
										TsdfGrid<VertexT, BoxT, TsdfT>* lastGrid, bool isVoxelsize)
	: HashGrid<VertexT, BoxT>(cellSize, bb, isVoxelsize)
{
	cout << timestamp << "Started creating grid " << "Values: " << size << endl;
	// get fusion slice from old grid if it exists one
	if(lastGrid != NULL)
	{
		this->m_queryPoints = lastGrid->getFusionPoints();
		this->m_qpIndices = lastGrid->getFusionIndices();
		this->m_cells = lastGrid->getFusionCells();
	}
	int center_of_bb_x = (this->m_boundingBox.getXSize()/2) / this->m_voxelsize;
	int center_of_bb_y = (this->m_boundingBox.getYSize()/2) / this->m_voxelsize;
	int	center_of_bb_z = (this->m_boundingBox.getZSize()/2) / this->m_voxelsize;
	m_fusionIndex_x = shiftX + center_of_bb_x;
	m_fusionIndex_y = shiftY + center_of_bb_y;
	m_fusionIndex_z = shiftZ + center_of_bb_z;
	//#pragma omp parallllell for
	m_fusionIndex = 0;
	size_t last_size = this->m_queryPoints.size();
	this->m_queryPoints.resize(size + last_size);
	for(size_t i = 0; i < size; i++)
	{
		int grid_index = i + last_size;
		// shift tsdf into global grid
		int global_x = tsdf[i].x + center_of_bb_x;
		int global_y = tsdf[i].y + center_of_bb_y;
		int global_z = tsdf[i].z + center_of_bb_z;
		VertexT position(global_x, global_y, global_z);
		QueryPoint<VertexT> qp = QueryPoint<VertexT>(position, tsdf[i].w);
		this->m_queryPoints[grid_index] = qp;
		size_t hash_value = this->hashValue(global_x, global_y, global_z);
		this->m_qpIndices[hash_value] = grid_index;
		this->m_globalIndex++;
		if((global_x == m_fusionIndex_x) || (global_y == m_fusionIndex_y) || (global_z == m_fusionIndex_z))
		{
			m_fusion_qpIndices[hash_value] = m_fusionIndex;
			m_fusionPoints.push_back(qp);
			m_fusionIndex++;
		}
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
	cout << "Fusion Boxes " << m_fusion_cells.size() << endl;
}

template<typename VertexT, typename BoxT, typename TsdfT>
void TsdfGrid<VertexT, BoxT, TsdfT>::addTSDFLatticePoint(int index_x, int index_y, int index_z, float distance)
{
	bool isFusion = false;
	if((index_x == m_fusionIndex_x) || (index_y == m_fusionIndex_y) || (index_z == m_fusionIndex_z))
	{
		isFusion = true;
	}
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

		it = this->m_cells.find(hash_value);
		if(it == this->m_cells.end())
		{
			//Calculate box center .. useless
			VertexT box_center(
					(index_x),
					(index_y),
					(index_z));

			//Create new box
			BoxT* box = new BoxT(box_center, isFusion);
			vector<size_t> boxQps;
			boxQps.resize(8);
			vector<size_t> cornerHashs;
			cornerHashs.resize(8);
			//Setup the box itself
			for(int k = 0; k < 8; k++)
			{
				//Find point in Grid
				dx = TSDFCreateTable[k][0];
				dy = TSDFCreateTable[k][1];
				dz = TSDFCreateTable[k][2];
				size_t corner_hash = this->hashValue(index_x + dx, index_y + dy, index_z + dz);
				if(!isFusion && ((index_x + dx == m_fusionIndex_x) || (index_y + dy == m_fusionIndex_y) || (index_z + dz == m_fusionIndex_z)))
				{
					isFusion = true;
					box->setFusion(true);
				}
				auto qp_index_it = this->m_qpIndices.find(corner_hash);
				//If point exist, save index in box
				if(qp_index_it != this->m_qpIndices.end()) 
				{
					box->setVertex(k, qp_index_it->second);
				}
				else
				{
					delete box;
					return;
				}
				boxQps[k] = qp_index_it->second;
				cornerHashs[k] = corner_hash;
			}
			if(isFusion)
			{
				for(size_t i = 0; i < 8; i++)
				{
					m_fusion_qpIndices[cornerHashs[i]] = m_fusionIndex;
					m_fusionPoints.push_back(this->m_queryPoints[boxQps[i]]);
					box->setVertex(i, m_fusionIndex);
					m_fusionIndex++;
				}
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
						//cout << "neigbour index " << neighbor_index << endl;
						neighbor_index++;
					}
				}
			}

			this->m_cells[hash_value] = box;
			if(isFusion)
			{
				this->m_fusion_cells[hash_value] = box;
			}
		}
		else
		{
			cout << "double cell " << endl;
			//cout << "index z: " << index_z <<  endl;
		}
	//}

}

template<typename VertexT, typename BoxT, typename TsdfT>
TsdfGrid<VertexT, BoxT, TsdfT>::~TsdfGrid()
{

}

} /* namespace lvr */
