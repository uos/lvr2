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
 * FastBox.cpp
 *
 *  Created on: 03.03.2011
 *      Author: Thomas Wiemann
 */


namespace lvr
{

template<typename VertexT, typename NormalT>
float FastBox<VertexT, NormalT>::m_voxelsize = 0;

template<typename VertexT, typename NormalT>
uint FastBox<VertexT, NormalT>::INVALID_INDEX = numeric_limits<uint>::max();

template<typename VertexT, typename NormalT>
FastBox<VertexT, NormalT>::FastBox(VertexT &center, bool fusionBox): m_fusionBox(fusionBox)
{
    // Init members
    for(int i = 0; i < 12; i++)
    {
    	m_intersections[i] = INVALID_INDEX;
    }

    for(int i = 0; i < 8; i++)
    {
    	m_vertices[i] = INVALID_INDEX;
    }

    for(int i = 0; i < 27; i++)
    {
        m_neighbors[i] = 0;
    }

    m_center = center;
}

template<typename VertexT, typename NormalT>
void FastBox<VertexT, NormalT>::setVertex(int index, uint nb)
{
    m_vertices[index] = nb;
}

template<typename VertexT, typename NormalT>
void FastBox<VertexT, NormalT>::setFusion(bool fusionBox)
{
    m_fusionBox = fusionBox;
}

template<typename VertexT, typename NormalT>
void FastBox<VertexT, NormalT>::setNeighbor(int index, FastBox<VertexT, NormalT>* nb)
{
    m_neighbors[index] = nb;
}


template<typename VertexT, typename NormalT>
FastBox<VertexT, NormalT>* FastBox<VertexT, NormalT>::getNeighbor(int index)
{
    return m_neighbors[index];
}

template<typename VertexT, typename NormalT>
uint FastBox<VertexT, NormalT>::getVertex(int index)
{
    return m_vertices[index];
}



template<typename VertexT, typename NormalT>
void FastBox<VertexT, NormalT>::getCorners(VertexT corners[],
                                           vector<QueryPoint<VertexT> > &qp)
{
    // Get the box corner positions from the query point array
    for(int i = 0; i < 8; i++)
    {
        corners[i] = VertexT(qp[m_vertices[i]].m_position);
    }
}

template<typename VertexT, typename NormalT>
void FastBox<VertexT, NormalT>::getDistances(float distances[],
                                             vector<QueryPoint<VertexT> > &qp)
{
    // Get the distance values from the query point array
    // for the corners of the current box
    for(int i = 0; i < 8; i++)
    {
        distances[i] = qp[m_vertices[i]].m_distance;
    }
}

template<typename VertexT, typename NormalT>
int  FastBox<VertexT, NormalT>::getIndex(vector<QueryPoint<VertexT> > &qp)
{
    // Determine the MC-Table index for the current corner configuration
    int index = 0;
    for(int i = 0; i < 8; i++)
    {
        if(qp[m_vertices[i]].m_distance > 0) index |= (1 << i);
    }
    return index;
}

template<typename VertexT, typename NormalT>
float FastBox<VertexT, NormalT>::calcIntersection(float x1, float x2, float d1, float d2)
{

    // Calculate the surface intersection using linear interpolation
      // and check for different signs of the given distance values.
      // If for some reason there was no sign change, return the
      // middle point
      if( (d1 < 0 && d2 >= 0) || (d2 < 0 && d1 >= 0) )
      {
          return  x2 - d2 * (x1 - x2) / (d1 - d2);
      }
      else
      {
          return  (x2 + x1) / 2.0;
      }
}

template<typename VertexT, typename NormalT>
void FastBox<VertexT, NormalT>::getIntersections(VertexT* corners,
                                                 float* distance,
                                                 VertexT* positions)
{
    //float d1, d2;
    //d1 = 
    //d2 = 0;

    float intersection;

	VertexT v1 = corners[0];

    intersection = calcIntersection( (corners[0])[0], (corners[1])[0], distance[0], distance[1]);
    positions[0] = VertexT(intersection, corners[0][1], corners[0][2]);

    intersection = calcIntersection(corners[1][1], corners[2][1], distance[1], distance[2]);
    positions[1] = VertexT(corners[1][0], intersection, corners[1][2]);

    intersection = calcIntersection(corners[3][0], corners[2][0], distance[3], distance[2]);
    positions[2] = VertexT(intersection, corners[2][1], corners[2][2]);

    intersection = calcIntersection(corners[0][1], corners[3][1], distance[0], distance[3]);
    positions[3] = VertexT(corners[3][0], intersection, corners[3][2]);

    //Back Quad
    intersection = calcIntersection(corners[4][0], corners[5][0], distance[4], distance[5]);
    positions[4] = VertexT(intersection, corners[4][1], corners[4][2]);

    intersection = calcIntersection(corners[5][1], corners[6][1], distance[5], distance[6]);
    positions[5] = VertexT(corners[5][0], intersection, corners[5][2]);


    intersection = calcIntersection(corners[7][0], corners[6][0], distance[7], distance[6]);
    positions[6] = VertexT(intersection, corners[6][1], corners[6][2]);

    intersection = calcIntersection(corners[4][1], corners[7][1], distance[4], distance[7]);
    positions[7] = VertexT(corners[7][0], intersection, corners[7][2]);

    //Sides
    intersection = calcIntersection(corners[0][2], corners[4][2], distance[0], distance[4]);
    positions[8] = VertexT(corners[0][0], corners[0][1], intersection);

    intersection = calcIntersection(corners[1][2], corners[5][2], distance[1], distance[5]);
    positions[9] = VertexT(corners[1][0], corners[1][1], intersection);

    intersection = calcIntersection(corners[3][2], corners[7][2], distance[3], distance[7]);
    positions[10] = VertexT(corners[3][0], corners[3][1], intersection);

    intersection = calcIntersection(corners[2][2], corners[6][2], distance[2], distance[6]);
    positions[11] = VertexT(corners[2][0], corners[2][1], intersection);

}


template<typename VertexT, typename NormalT>
void FastBox<VertexT, NormalT>::getSurface(BaseMesh<VertexT, NormalT> &mesh,
                                               vector<QueryPoint<VertexT> > &qp,
                                               uint &globalIndex)
{
	if(!m_fusionBox)
    {
		VertexT corners[8];
		VertexT vertex_positions[12];

		float distances[8];

		getCorners(corners, qp);
		getDistances(distances, qp);
		getIntersections(corners, distances, vertex_positions);

		int index = getIndex(qp);

		// Do not create traingles for invalid boxes
		for (int i = 0; i < 8; i++)
		{
			if (qp[m_vertices[i]].m_invalid)
			{
				return;
			}
		}

		uint edge_index = 0;

		int triangle_indices[3];
		// Generate the local approximation sirface according to the marching
		// cubes table for Paul Burke.
		for(int a = 0; MCTable[index][a] != -1; a+= 3){
			for(int b = 0; b < 3; b++){
				edge_index = MCTable[index][a + b];

				//If no index was found generate new index and vertex
				//and update all neighbor boxes
				if(m_intersections[edge_index] == INVALID_INDEX)
				{
					m_intersections[edge_index] = globalIndex;
					VertexT v = vertex_positions[edge_index];

					// Insert vertex and a new temp normal into mesh.
					// The normal is inserted to assure that vertex
					// and normal array always have the same size.
					// The actual normal is interpolated later.
					mesh.addVertex(v);
					mesh.addNormal(NormalT());
					for(int i = 0; i < 3; i++)
					{
						FastBox<VertexT, NormalT>* current_neighbor = m_neighbors[neighbor_table[edge_index][i]];
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
				triangle_indices[b] = m_intersections[edge_index];
			}

			// Add triangle actually does the normal interpolation for us.
			mesh.addTriangle(triangle_indices[0], triangle_indices[1], triangle_indices[2]);
		}
	}
	else
	{
		m_fusionBox = false;
	}
}

} // namespace lvr
