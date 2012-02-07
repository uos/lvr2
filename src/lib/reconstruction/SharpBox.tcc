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

//TODO: set in main()
template<typename VertexT, typename NormalT>
float SharpBox<VertexT, NormalT>::m_theta_sharp = 0.9;

template<typename VertexT, typename NormalT>
float SharpBox<VertexT, NormalT>::m_phi_corner = 0.7;

template<typename VertexT, typename NormalT>
SharpBox<VertexT, NormalT>::SharpBox(VertexT v, typename PointsetSurface<VertexT>::Ptr surface) : FastBox<VertexT, NormalT>(v)
{
	m_surface = surface;
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
void SharpBox<VertexT, NormalT>::getSurface(
        BaseMesh<VertexT, NormalT> &mesh,
        vector<QueryPoint<VertexT> > &query_points,
        uint &globalIndex)
{
    typedef SharpBox<VertexT, NormalT>*  p_sBox;
    VertexT corners[8];
    VertexT vertex_positions[12];
    NormalT vertex_normals[12];

    float distances[8];

    getCorners(corners, query_points);
    getDistances(distances, query_points);
    getIntersections(corners, distances, vertex_positions);
    getNormals(vertex_positions, vertex_normals);

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
    bool containsSharpFeature = false;
    bool containsSharpCorner = false;
    NormalT n_asterisk;
    float phi = -FLT_MAX;
    //TODO: was passiert für ungültige indizes?
    for (int i = 0; i < 12; i++)
    {
    	for (int j = 0; j < 12; j++)
    	{
    		if (i != j)
    		{
    			//save n_i x n_j if they enclose the largest angle
    			if(vertex_normals[i] * vertex_normals[j] > phi)
    			{
    				phi = vertex_normals[i] * vertex_normals[j];
    				n_asterisk = vertex_normals[i].cross(vertex_normals[j]);
    			}
    			if (vertex_normals[i] * vertex_normals[j] < m_theta_sharp)
    			{
    				containsSharpFeature = true;
    			}
    		}
    	}
    }
    // Check for presence of sharp corners
    if (containsSharpFeature)
    {
    	for (int i = 0; i < 12; i++)
    	{
    		if (fabs(vertex_normals[i] * n_asterisk) > m_phi_corner)
    		{
    			containsSharpCorner = true;
    		}
    	}
    }

    // Sharp feature detected -> use extended marching cubes
    if (containsSharpFeature)
    {
    	//TODO: Solve LGS
    	//Insert point and triangle fan
    }
    else     // No sharp features present -> use standard marching cubes
    {
    	uint edge_index = 0;

    	int triangle_indices[3];

    	// Generate the local approximation sirface according to the marching
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

    		// Add triangle actually does the normal interpolation for us.
    		mesh.addTriangle(triangle_indices[0], triangle_indices[1], triangle_indices[2]);
    	}
    }
}


} /* namespace lssr */
