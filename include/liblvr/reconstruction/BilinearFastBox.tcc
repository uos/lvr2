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
 * BilinearFastBox.tcc
 *
 *  @date 16.02.2012
 *  @author Thomas Wiemann
 */

#include "FastBox.hpp"

namespace lvr
{

template<typename VertexT, typename NormalT>
typename PointsetSurface<VertexT>::Ptr BilinearFastBox<VertexT, NormalT>::m_surface(0);


/*static int PlaneTable[28] =
{
        204, 51, 153, 102, 240, 15,
        192, 63, 48, 207, 12, 243, 3, 252, 96, 144, 6, 9, 159, 111, 249, 246, 136, 119, 17, 238, 34, 221
};*/

template<typename VertexT, typename NormalT>
const string BoxTraits<BilinearFastBox<VertexT, NormalT> >::type = "BilinearFastBox";


template<typename VertexT, typename NormalT>
BilinearFastBox<VertexT, NormalT>::BilinearFastBox(VertexT &center)
    : FastBox<VertexT, NormalT>(center), m_mcIndex(0)
{

}

template<typename VertexT, typename NormalT>
void BilinearFastBox<VertexT, NormalT>::getSurface(
        BaseMesh<VertexT, NormalT> &m,
        vector<QueryPoint<VertexT> > &qp,
        uint &globalIndex)
{
    // Cast mesh type
    HalfEdgeMesh<VertexT, NormalT> *mesh;
    mesh = static_cast<HalfEdgeMesh<VertexT, NormalT>* >(&m);

    VertexT corners[8];
    VertexT vertex_positions[12];

    float distances[8];

    this->getCorners(corners, qp);
    this->getDistances(distances, qp);
    this->getIntersections(corners, distances, vertex_positions);

    int index = this->getIndex(qp);
    m_mcIndex = index;

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
                mesh->addVertex(v);
                mesh->addNormal(NormalT());
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
        HalfEdgeFace<VertexT, NormalT>* f;
        mesh->addTriangle(triangle_indices[0], triangle_indices[1], triangle_indices[2], f);
        m_faces.push_back(f);
    }
}

template<typename VertexT, typename NormalT>
void BilinearFastBox<VertexT, NormalT>::optimizePlanarFaces(size_t kc)
{
	if(this->m_surface)
	{
		typedef HalfEdge<HalfEdgeVertex<VertexT, NormalT>, HalfEdgeFace<VertexT, NormalT> > HEdge;

		typename SearchTree<VertexT>::Ptr tree = this->m_surface->searchTree();

		// Detect triangles that are on the border of the mesh
		vector<HEdge*> out_edges;

		for(int i = 0; i < m_faces.size(); i++)
		{
			HalfEdgeFace<VertexT, NormalT>* face = m_faces[i];
			HEdge* e = face->m_edge;
			for(int j = 0; j < 2; j++)
			{
				// Catch null pointer from outer faces
				try
				{
					e->pair()->face();
				}
				catch (HalfEdgeAccessException)
				{
					out_edges.push_back(e);
				}

				// Check integrety
				try
				{
					e = e->next();
				}
				catch (HalfEdgeAccessException)
				{
					// Face currupted, abort
					cout << "Warning, currupted face" << endl;
					break;
				}
			}

		}

		// Handle different cases
		if(out_edges.size() == 1 || out_edges.size() == 2 )
		{
			// Get nearest points
			for(int i = 0; i < out_edges.size(); i++)
			{

				vector<VertexT> nearest1, nearest2;
				tree->kSearch( out_edges[i]->start()->m_position, kc, nearest1);

				size_t nk = min(kc, nearest1.size());


				// Hmmm, sometimes the k-search seems to fail...
				if(nk > 0)
				{
					VertexT centroid1;
					for(int a = 0; a < nk; a++)
					{
						centroid1 += nearest1[a];
					}
					centroid1 /= nk;
					out_edges[i]->start()->m_position = centroid1;
				}

				tree->kSearch( out_edges[i]->end()->m_position, kc, nearest2);
				nk = min(kc, nearest2.size());

				if(nk > 0)
				{
					VertexT centroid2;
					for(int a = 0; a < nk; a++)
					{
						centroid2 += nearest2[a];
					}
					centroid2 /= nk;
					out_edges[i]->end()->m_position = centroid2;
				}

			}
		}
	}
}

template<typename VertexT, typename NormalT>
BilinearFastBox<VertexT, NormalT>::~BilinearFastBox()
{
    //for(int i = 0; i < m_faces.size(); i++) delete[] m_faces[i];
    m_faces.clear();
}

} /* namespace lvr */
