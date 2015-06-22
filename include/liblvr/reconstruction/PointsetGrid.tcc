/*
 * PointsetGrid.tcc
 *
 *  Created on: Nov 27, 2014
 *      Author: twiemann
 */

#include "PointsetGrid.hpp"

namespace lvr
{

template<typename VertexT, typename BoxT>
PointsetGrid<VertexT, BoxT>::PointsetGrid(float cellSize, typename PointsetSurface<VertexT>::Ptr& surface, BoundingBox<VertexT> bb, bool isVoxelsize)
	: HashGrid<VertexT, BoxT>(cellSize, bb, isVoxelsize), m_surface(surface)
{
	PointBufferPtr buffer = surface->pointBuffer();

	VertexT v_min = this->m_boundingBox.getMin();
	VertexT v_max = this->m_boundingBox.getMax();

	// Get indexed point buffer pointer
	size_t num_points;
	coord3fArr points = this->m_surface->pointBuffer()->getIndexedPointArray(num_points);

	size_t index_x, index_y, index_z;

	cout << timestamp << "Creating Grid..." << endl;

	// Iterator over all points, calc lattice indices and add lattice points to the grid
	for(size_t i = 0; i < num_points; i++)
	{
		index_x = calcIndex((points[i][0] - v_min[0]) / this->m_voxelsize);
		index_y = calcIndex((points[i][1] - v_min[1]) / this->m_voxelsize);
		index_z = calcIndex((points[i][2] - v_min[2]) / this->m_voxelsize);
		this->addLatticePoint(index_x, index_y, index_z);
	}

}

template<typename VertexT, typename BoxT>
void PointsetGrid<VertexT, BoxT>::addLatticePoint(int index_x, int index_y, int index_z, float distance)
{
	size_t hash_value;

	unsigned int INVALID = BoxT::INVALID_INDEX;

	float vsh = 0.5 * this->m_voxelsize;

	// Some iterators for hash map accesses
	typename HashGrid<VertexT, BoxT>::box_map_it it;
	typename HashGrid<VertexT, BoxT>::box_map_it neighbor_it;

	// Values for current and global indices. Current refers to a
	// already present query point, global index is id that the next
	// created query point will get
	unsigned int current_index = 0;

	int dx, dy, dz;

	// Get min and max vertex of the point clouds bounding box
	VertexT v_min = this->m_boundingBox.getMin();
	VertexT v_max = this->m_boundingBox.getMax();

	int e;
	this->m_extrude ? e = 8 : e = 1;
	for(int j = 0; j < e; j++)
	{
		// Get the grid offsets for the neighboring grid position
		// for the given box corner
		dx = HGCreateTable[j][0];
		dy = HGCreateTable[j][1];
		dz = HGCreateTable[j][2];

		hash_value = this->hashValue(index_x + dx, index_y + dy, index_z +dz);

		it = this->m_cells.find(hash_value);
		if(it == this->m_cells.end())
		{
			//Calculate box center
			VertexT box_center(
					(index_x + dx) * this->m_voxelsize + v_min[0],
					(index_y + dy) * this->m_voxelsize + v_min[1],
					(index_z + dz) * this->m_voxelsize + v_min[2]);

			//Create new box
			BoxT* box = new BoxT(box_center);

			//Setup the box itself
			for(int k = 0; k < 8; k++){

				//Find point in Grid
				current_index = this->findQueryPoint(k, index_x + dx, index_y + dy, index_z + dz);
				//If point exist, save index in box
				if(current_index != INVALID) box->setVertex(k, current_index);

				//Otherwise create new grid point and associate it with the current box
				else
				{
					VertexT position(box_center[0] + box_creation_table[k][0] * vsh,
							box_center[1] + box_creation_table[k][1] * vsh,
							box_center[2] + box_creation_table[k][2] * vsh);

					this->m_queryPoints.push_back(QueryPoint<VertexT>(position, distance));
					box->setVertex(k, this->m_globalIndex);
					this->m_globalIndex++;

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
						neighbor_hash = this->hashValue(index_x + dx + a,
												  	    index_y + dy + b,
														index_z + dz + c);

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
		}
	}

}

template<typename VertexT, typename BoxT>
void PointsetGrid<VertexT, BoxT>::calcDistanceValues()
{
	// Status message output
	string comment = timestamp.getElapsedTime() + "Calculating distance values ";
	ProgressBar progress(this->m_queryPoints.size(), comment);

	Timestamp ts;

	// Calculate a distance value for each query point
	#pragma omp parallel for
	for( int i = 0; i < (int)this->m_queryPoints.size(); i++){
		float projectedDistance;
		float euklideanDistance;

		//cout << euklideanDistance << " " << projectedDistance << endl;

		this->m_surface->distance(this->m_queryPoints[i].m_position, projectedDistance, euklideanDistance);
		if (euklideanDistance > 1.7320 * this->m_voxelsize)
		{
			this->m_queryPoints[i].m_invalid = true;
		}
		this->m_queryPoints[i].m_distance = projectedDistance;
		++progress;
	}
	cout << endl;
	cout << timestamp << "Elapsed time: " << ts << endl;
}

template<typename VertexT, typename BoxT>
PointsetGrid<VertexT, BoxT>::~PointsetGrid()
{
	// TODO Auto-generated destructor stub
}

} /* namespace lvr */
