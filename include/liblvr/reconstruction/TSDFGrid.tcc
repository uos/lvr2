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
		this->m_queryPoints = vector<QueryPoint<VertexT> >(lastGrid->getFusionPoints());
		this->m_qpIndices = qp_map(lastGrid->getFusionIndices());
		this->m_cells = lastGrid->getFusionCells();
		/*for(auto cellPair : lastGrid->getFusionCells())
		{
			BoxT* box = new BoxT(*(cellPair.second));
			this->m_cells[cellPair.first] = cellPair.second;
		}*/
		//this->m_global_cells = lastGrid->m_global_cells;
	}
	int center_of_bb_x = (this->m_boundingBox.getXSize()/2) / this->m_voxelsize;
	int center_of_bb_y = (this->m_boundingBox.getYSize()/2) / this->m_voxelsize;
	int	center_of_bb_z = (this->m_boundingBox.getZSize()/2) / this->m_voxelsize;
	m_fusionIndex_x = shiftX + center_of_bb_x;
	m_fusionIndex_y = shiftY + center_of_bb_y;
	m_fusionIndex_z = shiftZ + center_of_bb_z;
	//#pragma omp parallllell for
	m_fusionIndex = 0;
	int grid_index = 0;
	size_t last_size = this->m_queryPoints.size();
	this->m_queryPoints.resize(size + last_size);
	for(size_t i = 0; i < size; i++)
	{
		grid_index = i + last_size;
		// shift tsdf into global grid
		int global_x = tsdf[i].x + center_of_bb_x;
		int global_y = tsdf[i].y + center_of_bb_y;
		int global_z = tsdf[i].z + center_of_bb_z;
		VertexT position(global_x, global_y, global_z);
		QueryPoint<VertexT> qp = QueryPoint<VertexT>(position, tsdf[i].w);
		this->m_queryPoints[grid_index] = qp;
		size_t hash_value = this->hashValue(global_x, global_y, global_z);
		this->m_qpIndices[hash_value] = grid_index;
	}
	this->m_globalIndex = grid_index + 1;
	cout << timestamp << "Finished inserting" << endl;
	// Iterator over all points, calc lattice indices and add lattice points to the grid
	for(size_t i = 0; i < size; i++)
	{
		int global_x = tsdf[i].x + center_of_bb_x;
		int global_y = tsdf[i].y + center_of_bb_y;
		int global_z = tsdf[i].z + center_of_bb_z;
		//#pragma omp task
		addLatticePoint(global_x , global_y, global_z, tsdf[i].w);
	}
	cout << timestamp << "Finished creating grid" << endl;
	//cout << "Repaired " << this->m_globalIndex - grid_index - 1 << " boxes " << endl;
	cout << "Fusion Boxes " << m_fusion_cells.size() << endl;
}

template<typename VertexT, typename BoxT, typename TsdfT>
void TsdfGrid<VertexT, BoxT, TsdfT>::addLatticePoint(int index_x, int index_y, int index_z, float distance)
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

	hash_value = this->hashValue(index_x, index_y, index_z);

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
	std::vector<int> missingCorner;
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
			boxQps[k] = qp_index_it->second;
		}
		else
		{
			delete box;
			return;
			//missingCorner.push_back(k);
			//boxQps[k] = 0;
		}
		cornerHashs[k] = corner_hash;
	}
	//if(missingCorner.size() > 0)
	//{
		/*for(int t = 0; t < missingCorner.size(); t++)
		{
			if(!repairCell(box, index_x, index_y, index_z, missingCorner[t], boxQps))
				return;
		}*/
		
	//}
	
	// add box to global cell map
	/*auto global_box = this->m_global_cells.find(hash_value);
	if(global_box != m_global_cells.end())
	{
		//delete box->m_intersections;
		//box->m_intersections = global_box->second;
		//cout << "double double " << endl;
		//box->m_doubleBox = true;
		delete box;
		return;
	}
	else
	{
		m_global_cells[hash_value] = box->m_intersections;
	}*/
	
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

template<typename VertexT, typename BoxT, typename TsdfT>
int TsdfGrid<VertexT, BoxT, TsdfT>::repairCell(BoxT* box, 
				int index_x, int index_y, int index_z, int corner, vector<size_t>& boxQps)
{
	//Find point in Grid
	int nx = box_creation_table[corner][0];
	int ny = box_creation_table[corner][1];
	int nz = box_creation_table[corner][2];
	//Find point in Grid
	int dx = TSDFCreateTable[corner][0];
	int dy = TSDFCreateTable[corner][1];
	int dz = TSDFCreateTable[corner][2];
	std::vector<size_t> neighbour_hashes;
	neighbour_hashes.resize(3);
	neighbour_hashes[0] = this->hashValue((index_x + dx) + nx, (index_y + dy) ,      (index_z + dz));
	neighbour_hashes[1] = this->hashValue((index_x + dx) ,     (index_y + dy) + ny , (index_z + dz));
	neighbour_hashes[2] = this->hashValue((index_x + dx) ,     (index_y + dy) ,      (index_z + dz) + nz);
	for(int j = 0; j < 3 ; j++)
	{
		//cout << "check interference " << endl;
		auto qp_index_it = this->m_qpIndices.find(neighbour_hashes[j]);
		//If point exist, interfere tsdf value and create new qp
		if(qp_index_it != this->m_qpIndices.end()) 
		{
			double tsdf_1 = this->m_queryPoints[qp_index_it->second].m_distance;
			int corner2 = box_neighbour_table[corner][j];
			double tsdf_2 = this->m_queryPoints[boxQps[corner2]].m_distance;
			double tsdf = (tsdf_1 + tsdf_2)/2;
			VertexT position(index_x + dx, index_y + dy, index_z + dz);
			QueryPoint<VertexT> qp = QueryPoint<VertexT>(position, tsdf);
			this->m_queryPoints.resize(this->m_queryPoints.size() + 1);
			this->m_queryPoints[this->m_globalIndex] = qp;
			size_t miss_hash = this->hashValue(index_x + dx, index_y + dy, index_z + dz);
			this->m_qpIndices[miss_hash] = this->m_globalIndex;
			box->setVertex(corner, this->m_globalIndex);
			boxQps[corner] = this->m_globalIndex;
			this->m_globalIndex++;
			return 1;
		}
	}
	delete box;
	return 0;
}

template<typename VertexT, typename BoxT, typename TsdfT>
TsdfGrid<VertexT, BoxT, TsdfT>::~TsdfGrid()
{
	box_map_it iter;
	for(iter = this->m_cells.begin(); iter != this->m_cells.end(); iter++)
	{
		if(iter->second != NULL)
		{
			if(!iter->second->m_fusedBox)
			{
				delete (iter->second);
				iter->second = NULL;
			}
			else
				iter->second->m_fusedBox = false;
		}
	}
}

} /* namespace lvr */
