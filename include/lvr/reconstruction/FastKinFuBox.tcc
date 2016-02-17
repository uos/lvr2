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
 * FastKinFuBox.cpp
 *
 *  Created on: 12.12.2015
 *      Author: Tristan Igelbrink
 */


namespace lvr
{
template<typename VertexT, typename NormalT>
FastKinFuBox<VertexT, NormalT>::FastKinFuBox(VertexT &center, bool fusionBox, bool oldFusionBox)
			: m_fusionBox(fusionBox), m_oldfusionBox(oldFusionBox), FastBox<VertexT, NormalT >(center)
{

}

template<typename VertexT, typename NormalT>
void FastKinFuBox<VertexT, NormalT>::setFusion(bool fusionBox)
{
    m_fusionBox = fusionBox;
}

template<typename VertexT, typename NormalT>
void FastKinFuBox<VertexT, NormalT>::getSurface(BaseMesh<VertexT, NormalT> &mesh,
                                               vector<QueryPoint<VertexT> > &qp,
                                               uint &globalIndex)
{
	VertexT corners[8];
	VertexT vertex_positions[12];

	float distances[8];

	this->getCorners(corners, qp);
	this->getDistances(distances, qp);
	this->getIntersections(corners, distances, vertex_positions);

	int index = this->getIndex(qp);

	// Do not create traingles for invalid boxes
	for (int i = 0; i < 8; i++)
	{
		if (qp[this->m_vertices[i]].m_invalid)
		{
			return;
		}
	}

	uint edge_index = 0;

	int triangle_indices[3];
	// Generate the local approximation surface according to the marching
	// cubes table for Paul Burke.
	for(int a = 0; MCTable[index][a] != -1; a+= 3){
		for(int b = 0; b < 3; b++){
			edge_index = MCTable[index][a + b];

			//If no index was found generate new index and vertex
			//and update all neighbor boxes
			if(this->m_intersections[edge_index] == this->INVALID_INDEX)
			{
				this->m_intersections[edge_index] = globalIndex;
				VertexT v = vertex_positions[edge_index];
				// Insert vertex and a new temp normal into mesh.
				// The normal is inserted to assure that vertex
				// and normal array always have the same size.
				// The actual normal is interpolated later.
				mesh.addVertex(v);
				mesh.addNormal(NormalT());
				int neighbour_count = 0;
				for(int i = 0; i < 3; i++)
				{
					FastKinFuBox<VertexT, NormalT>* current_neighbor = dynamic_cast< FastKinFuBox<VertexT, NormalT>* >(this->m_neighbors[neighbor_table[edge_index][i]]);
					if(current_neighbor != 0)
					{
						current_neighbor->m_intersections[neighbor_vertex_table[edge_index][i]] = globalIndex;
						if(!current_neighbor->m_fusionNeighborBox)
							neighbour_count++;
					}
				}
				if(m_fusionBox && neighbour_count < 3)
					dynamic_cast<HalfEdgeKinFuMesh<VertexT, NormalT>& >(mesh).setFusionVertex(globalIndex);
				if(m_oldfusionBox && neighbour_count < 3)
					dynamic_cast<HalfEdgeKinFuMesh<VertexT, NormalT>& >(mesh).setFusionNeighborVertex(globalIndex);
				// Increase the global vertex counter to save the buffer
				// position were the next new vertex has to be inserted
				globalIndex++;
			}

			//Save vertex index in mesh
			triangle_indices[b] = this->m_intersections[edge_index];
		}

		// Add triangle actually does the normal interpolation for us.
		mesh.addTriangle(triangle_indices[0], triangle_indices[1], triangle_indices[2]);
	}
	if(m_fusionBox)
	{
		m_fusionBox = false;
		m_fusedBox = true;
	}
	//if(m_oldfusionBox)
		//m_oldfusionBox = false;
}

} // namespace lvr
