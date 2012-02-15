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

/**
 * SharpBox.tcc
 *
 *  @date 06.02.2012
 *  @author Kim Rinnewitz (krinnewitz@uos.de)
 *  @author Sven Schalk (sschalk@uos.de)
 */

namespace lssr
{

template<typename VertexT, typename NormalT>
float SharpBox<VertexT, NormalT>::m_theta_sharp = 0.9;

template<typename VertexT, typename NormalT>
float SharpBox<VertexT, NormalT>::m_phi_corner = 0.7;

template<typename VertexT, typename NormalT>
SharpBox<VertexT, NormalT>::SharpBox(VertexT v, typename PointsetSurface<VertexT>::Ptr surface) : FastBox<VertexT, NormalT>(v)
{
	m_surface = surface;
	m_containsSharpFeature = false;
	m_containsSharpCorner = false;
}

template<typename VertexT, typename NormalT>
SharpBox<VertexT, NormalT>::~SharpBox()
{

}

template<typename VertexT, typename NormalT>
void SharpBox<VertexT, NormalT>::getNormals(VertexT vertex_positions[], NormalT vertex_normals[])
{
	for (int i = 0; i < 12; i++)
	{
		vertex_normals[i] = (NormalT) m_surface->getInterpolatedNormal(vertex_positions[i]);
	}
}

template<typename VertexT, typename NormalT>
void SharpBox<VertexT, NormalT>::detectSharpFeatures(VertexT vertex_positions[], NormalT vertex_normals[], uint index)
{
	//  skip unhandled configurations
	if (ExtendedMCTable[index][0] == -1)
	{
		m_containsSharpCorner = m_containsSharpFeature = false;
		return;
	}

	getNormals(vertex_positions, vertex_normals);

	NormalT n_asterisk;
	float phi = FLT_MAX;

	int edge_index1, edge_index2;
	for(int a = 0; MCTable[index][a] != -1; a+= 3)
	{
		for(int b = 0; b < 3; b++)
		{
			edge_index1 = MCTable[index][a + b];
			for(int c = 0; MCTable[index][c] != -1; c+= 3)
			{
				for(int d = 0; d < 3; d++)
				{
					edge_index2 = MCTable[index][c + d];
					if (edge_index1 != edge_index2)
					{
						//save n_i x n_j if they enclose the largest angle
						if(vertex_normals[edge_index1] * vertex_normals[edge_index2] < phi)
						{
							phi = vertex_normals[edge_index1] * vertex_normals[edge_index2];
							n_asterisk = vertex_normals[edge_index1].cross(vertex_normals[edge_index2]);
						}
						if (vertex_normals[edge_index1] * vertex_normals[edge_index2] < m_theta_sharp)
						{
							m_containsSharpFeature = true;
						}
					}
				}
			}
		}
	}

	// Check for presence of sharp corners
	if (m_containsSharpFeature)
	{
		for(int a = 0; MCTable[index][a] != -1; a+= 3)
		{
			for(int b = 0; b < 3; b++)
			{
				edge_index1 = MCTable[index][a + b];
				if (fabs(vertex_normals[edge_index1] * n_asterisk) > m_phi_corner)
				{
					m_containsSharpCorner = true;
				}
			}
		}
	}

	// Check for inconsistencies
	if(        index == 1   || index == 2   || index == 4   || index == 8		//corners
			|| index == 16  || index == 32  || index == 64  || index == 128
			|| index == 254 || index == 253 || index == 251 || index == 247
			|| index == 239 || index == 223 || index == 191 || index == 127 )
	{
		if (m_containsSharpCorner == false) // contradiction -> use standard marching cubes
		{
			m_containsSharpCorner = m_containsSharpFeature = false;
		}
	}
	else
	{
		m_containsSharpCorner = false;
	}
}


template<typename VertexT, typename NormalT>
void SharpBox<VertexT, NormalT>::getSurface(
        BaseMesh<VertexT, NormalT> &mesh,
        vector<QueryPoint<VertexT> > &query_points,
        uint &globalIndex)
{
	VertexT corners[8];
	VertexT vertex_positions[12];
	NormalT vertex_normals[12];

	float distances[8];

	getCorners(corners, query_points);
	getDistances(distances, query_points);
	getIntersections(corners, distances, vertex_positions);

	int index = getIndex(query_points);

	// Do not create traingles for invalid boxes
	for (int i = 0; i < 8; i++)
	{
		if (query_points[this->m_vertices[i]].m_invalid)
		{
			return;
		}
	}

	// Check for presence of sharp features in the box
	this->detectSharpFeatures(vertex_positions, vertex_normals, index);

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
				for(int i = 0; i < 3; i++)
				{
					FastBox<VertexT, NormalT>* current_neighbor = this->m_neighbors[neighbor_table[edge_index][i]];
					if(current_neighbor != 0)
					{
						current_neighbor->m_intersections[neighbor_vertex_table[edge_index][i]] = globalIndex;
					}
				}
				// Increase the global vertex counter to save the buffer
				// position were the next new vertex has to be inserted
				globalIndex++;
			}

			//Save vertex index in mesh
			triangle_indices[b] = this->m_intersections[edge_index];
		}
		if (!m_containsSharpFeature) // No sharp features present -> use standard marching cubes
		{
			// Add triangle actually does the normal interpolation for us.
			mesh.addTriangle(triangle_indices[0], triangle_indices[1], triangle_indices[2]);
		}
	}

	// Sharp feature detected -> use extended marching cubes
	if (m_containsSharpFeature)
	{
		// save for edge flipping
		m_extendedMCIndex = index;
		//calculate intersection for the new vertex position
		VertexT v = this->m_center;
		if (m_containsSharpCorner)
		{
			//First plane
			VertexT v1 = vertex_positions[ExtendedMCTable[index][0]];
			NormalT n1 = vertex_normals[ExtendedMCTable[index][0]];

			//Second plane
			VertexT v2 = vertex_positions[ExtendedMCTable[index][1]];
			NormalT n2 = vertex_normals[ExtendedMCTable[index][1]];

			//Third plane
			VertexT v3 = vertex_positions[ExtendedMCTable[index][3]];
			NormalT n3 = vertex_normals[ExtendedMCTable[index][3]];

			//calculate intersection between plane 1 and 2
			if (fabs(n1 * n2) < 0.9)
			{
				float d1 = n1 * v1;
				float d2 = n2 * v2;

				VertexT direction = n1.cross(n2);

				float denom = direction * direction;
				VertexT x = ((n2 * d1 - n1 * d2).cross(direction)) * (1 / denom);

				//calculate intersection between plane 3 and the intersection line between plane 1 and 2
				float denom2 = n3 * direction;
				if(fabs(denom2) > 0.0001)
				{
					float d = n3 * v3;
					float t = (d - n3 * x) / (denom2);

					VertexT intersection = x + direction * t;

					v = intersection;
				}
			}
		}
		else
		{
			//First plane
			VertexT v1 = (vertex_positions[ExtendedMCTable[index][2]] + vertex_positions[ExtendedMCTable[index][3]]) * 0.5;
			NormalT n1 = (vertex_normals[ExtendedMCTable[index][2]] + vertex_normals[ExtendedMCTable[index][3]]) * 0.5;
			//Second plane
			VertexT v2 = (vertex_positions[ExtendedMCTable[index][6]] + vertex_positions[ExtendedMCTable[index][7]]) * 0.5;
			NormalT n2 = (vertex_normals[ExtendedMCTable[index][6]] + vertex_normals[ExtendedMCTable[index][7]]) * 0.5;

			//calculate intersection between plane 1 and 2
			if (fabs(n1 * n2) < 0.9)
			{
				float d1 = n1 * v1;
				float d2 = n2 * v2;

				VertexT direction = n1.cross(n2);

				float denom = direction * direction;
				VertexT x = ((n2 * d1 - n1 * d2).cross(direction)) * (1 / denom);

				// project center of the box onto intersection line of the two planes
				v = x + direction * (((v - x) * direction) / (direction.length() * direction.length()));
			}

		}

		mesh.addVertex(v);
		mesh.addNormal(NormalT());
		uint index_center = globalIndex++;
		// Add triangle actually does the normal interpolation for us.
		for(int a = 0; ExtendedMCTable[index][a] != -1; a+= 2)
		{
			mesh.addTriangle(this->m_intersections[ExtendedMCTable[index][a]], index_center, this->m_intersections[ExtendedMCTable[index][a+1]]);
		}

	}
}


} /* namespace lssr */
