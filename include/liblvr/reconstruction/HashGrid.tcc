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
#include "HashGrid.hpp"
#include "geometry/BaseMesh.hpp"
#include "FastReconstructionTables.hpp"
#include "SharpBox.hpp"
#include "io/Progress.hpp"

namespace lvr
{

template<typename VertexT, typename BoxT>
HashGrid<VertexT, BoxT>::HashGrid(float cellSize, BoundingBox<VertexT> boundingBox, bool isVoxelsize) :
	m_extrude(false),
	m_boundingBox(boundingBox),
	m_globalIndex(0)
{
	m_coordinateScales[0] = 1.0;
	m_coordinateScales[1] = 1.0;
	m_coordinateScales[2] = 1.0;

	if(!m_boundingBox.isValid())
	{
		cout << timestamp << "Waring: Malformed BoundingBox." << endl;
	}

	if(!isVoxelsize)
	{
		m_voxelsize = (float) m_boundingBox.getLongestSide() / cellSize;
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
void HashGrid<VertexT, BoxT>::setCoordinateScaling(float x, float y, float z)
{
	m_coordinateScales[0] = x;
	m_coordinateScales[1] = y;
	m_coordinateScales[2] = z;
}

template<typename VertexT, typename BoxT>
HashGrid<VertexT, BoxT>::~HashGrid()
{
	/*box_map_it iter;
	for(iter = m_cells.begin(); iter != m_cells.end(); iter++)
	{
		if(iter->second != NULL)
		{
			delete (iter->second);
			iter->second = NULL;
		}
	}*/
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
unsigned int HashGrid<VertexT, BoxT>::findQueryPoint(
		const int &position, const int &x, const int &y, const int &z)
{
	int n_x, n_y, n_z, q_v, offset;
	box_map_it it;

	for(int i = 0; i < 7; i++)
	{
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
		typename unordered_map<size_t, BoxT* >::iterator it;
		BoxT* box;
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
