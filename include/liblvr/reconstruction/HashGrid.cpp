/*
 * HashGrid.cpp
 *
 *  Created on: Nov 27, 2014
 *      Author: twiemann
 */



/* Copyright (C) 2011 Uni Osnabrück
 * This file is part of the LAS VEGAS Reconstruction Toolkit,
 *
 * LAS VEGAS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * LAS VEGAS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA
 */


/*
 * HashGrid.cpp
 *
 *  Created on: 16.02.2011
 *      Author: Thomas Wiemann
 */
#include "HashGrid.tcc"

#include "geometry/BaseMesh.hpp"
#include "HashGridTables.hpp"
#include "SharpBox.hpp"
#include "io/Progress.hpp"

namespace lvr
{

template<typename VertexT, typename BoxT>
HashGrid<VertexT, BoxT>::HashGrid(float cellSize, BoundingBox<VertexT> boundingBox, bool isVoxelsize) :
	m_extrude(true),
	m_boundingBox(boundingBox),
	m_globalIndex(0)
{

	if(!m_boundingBox.isValid())
	{
		cout << timestamp << "Waring: Malformed BoundingBox." << endl;
	}

	if(!isVoxelsize)
	{
		m_voxelsize = (float) m_boundingBox.getLongestSide() / resolution;
	}
	else
	{
		m_voxelsize = cellSize;
	}

	cout << timestamp << "Used voxelsize is " << m_voxelsize << endl;

	if(!m_extrude)
	{
		cout << timestamp << "Grid is not extruded." << endl;
	}

	BoxT::m_voxelsize = m_voxelsize;
	calcIndices();
}

template<typename VertexT, typename BoxT>
HashGrid<VertexT, BoxT>::~HashGrid()
{
	box_map_it iter;
	for(iter = m_cells.begin(); iter != m_cells.end(); iter++)
	{
		delete ((*iter).second);
	}

	m_cells.clear();
}



template<typename VertexT, typename BoxT>
void HashGrid<VertexT, BoxT>::calcIndices()
{
	float max_size = m_boundingBox.getLongestSide();

	//Save needed grid parameters
	m_maxIndex = (int)ceil( (max_size + 5 * m_voxelsize) / m_voxelsize);
	m_maxIndexSquare = m_maxIndex * m_maxIndex;

	m_maxIndexX = (int)ceil(m_boundingBox.getXSize() / m_voxelsize) + 1;
	m_maxIndexY = (int)ceil(m_boundingBox.getYSize() / m_voxelsize) + 2;
	m_maxIndexZ = (int)ceil(m_boundingBox.getZSize() / m_voxelsize) + 3;
}

template<typename VertexT, typename BoxT>
unsigned int HashGrid<VertexT, typename BoxT>::findQueryPoint(
		const int &position, const int &x, const int &y, const int &z)
		{
	int n_x, n_y, n_z, q_v, offset;
	box_map_it it;

	for(int i = 0; i < 7; i++){
		offset = i * 4;
		n_x = x + shared_vertex_table[position][offset];
		n_y = y + shared_vertex_table[position][offset + 1];
		n_z = z + shared_vertex_table[position][offset + 2];
		q_v = shared_vertex_table[position][offset + 3];

		size_t hash = hashValue(n_x, n_y, n_z);

		it = m_cells.find(hash);
		if(it != m_cells.end())
		{
			BoxT* b = it->second;
			if(b->getVertex(q_v) != BoxT::INVALID_INDEX) return b->getVertex(q_v);
		}
	}

	return BoxT::INVALID_INDEX;
		}

template<typename VertexT, typename BoxT>
void HashGrid<VertexT, BoxT>::addLatticePoint(size_t index_x, size_t index_y, size_t index_z, float distance = 0.0)
{
	size_t hash_value;

	unsigned int INVALID = FastBox<VertexT, BoxT>::INVALID_INDEX;

	float vsh = 0.5 * m_voxelsize;

	// Some iterators for hash map accesses
	box_map_it it;
	box_map_it neighbor_it;

	// Values for current and global indices. Current refers to a
	// already present query point, global index is id that the next
	// created query point will get
	unsigned int current_index = 0;

	int dx, dy, dz;

	// Get min and max vertex of the point clouds bounding box
	VertexT v_min = m_boundingBox.getMin();
	VertexT v_max = m_boundingBox.getMax();

	int e;
	m_extrude ? e = 8 : e = 1;

	// Get the grid offsets for the neighboring grid position
	// for the given box corner
	dx = HGCreateTable[j][0];
	dy = HGCreateTable[j][1];
	dz = HGCreateTable[j][2];

	hash_value = hashValue(index_x + dx, index_y + dy, index_z +dz);


	it = m_cells.find(hash_value);
	if(it == m_cells.end())
	{
		//Calculate box center
		VertexT box_center(
				(index_x + dx) * m_voxelsize + v_min[0],
				(index_y + dy) * m_voxelsize + v_min[1],
				(index_z + dz) * m_voxelsize + v_min[2]);

		//Create new box
		BoxT* box = new BoxT(box_center);

		//Setup the box itself
		for(int k = 0; k < 8; k++){

			//Find point in Grid
			current_index = findQueryPoint(k, index_x + dx, index_y + dy, index_z + dz);

			//If point exist, save index in box
			if(current_index != INVALID) box->setVertex(k, current_index);

			//Otherwise create new grid point and associate it with the current box
			else
			{
				VertexT position(box_center[0] + box_creation_table[k][0] * vsh,
						box_center[1] + box_creation_table[k][1] * vsh,
						box_center[2] + box_creation_table[k][2] * vsh);

				m_queryPoints.push_back(QueryPoint<VertexT>(position));

				box->setVertex(k, m_globalIndex);
				m_globalIndex++;

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
					neighbor_hash = hashValue(index_x + dx + a,
							index_y + dy + b,
							index_z + dz + c);

					//Try to find this cell in the grid
					neighbor_it = m_cells.find(neighbor_hash);

					//If it exists, save pointer in box
					if(neighbor_it != m_cells.end())
					{
						box->setNeighbor(neighbor_index, (*neighbor_it).second);
						(*neighbor_it).second->setNeighbor(26 - neighbor_index, box);
					}

					neighbor_index++;
				}
			}
		}

		m_cells[hash_value] = box;
	}

}


template<typename VertexT, typename BoxT>
void HashGrid<VertexT, BoxT>::saveGrid(string filename)
{
	cout << timestamp << "Writing grid..." << endl;

	// Open file for writing
	ofstream out(filename.c_str());

	// Write data
	if(out.good())
	{
		// Write header
		out << m_queryPoints.size() << " " << m_voxelsize << " " << m_cells.size() << endl;

		// Write query points and distances
		for(size_t i = 0; i < m_queryPoints.size(); i++)
		{
			out << m_queryPoints[i].m_position[0] << " "
					<< m_queryPoints[i].m_position[1] << " "
					<< m_queryPoints[i].m_position[2] << " ";

			if(!isnan(m_queryPoints[i].m_distance))
			{
				out << m_queryPoints[i].m_distance << endl;
			}
			else
			{
				out << 0 << endl;
			}

		}

		// Write box definitions
		typename unordered_map<size_t, FastBox<VertexT, BoxT>* >::iterator it;
		FastBox<VertexT, BoxT>* box;
		for(it = m_cells.begin(); it != m_cells.end(); it++)
		{
			box = it->second;
			for(int i = 0; i < 8; i++)
			{
				out << box->getVertex(i) << " ";
			}
			out << endl;
		}
	}
}

} //namespace lvr
