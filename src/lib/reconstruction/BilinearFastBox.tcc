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

namespace lssr
{

template<typename VertexT, typename NormalT>
BilinearFastBox<VertexT, NormalT>::BilinearFastBox(VertexT &center)
    : FastBox<VertexT, NormalT>(center)
{

}

template<typename VertexT, typename NormalT>
void BilinearFastBox<VertexT, NormalT>::getSurface(
        HalfEdgeMesh<VertexT, NormalT> &mesh,
        vector<QueryPoint<VertexT> > &qp,
        uint &globalIndex)
{
    VertexT corners[8];
    VertexT vertex_positions[12];

    float distances[8];

    getCorners(corners, qp);
    getDistances(distances, qp);
    getIntersections(corners, distances, vertex_positions);

    int index = getIndex(qp);
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
        HalfEdgeFace<VertexT, NormalT>* f;
        mesh.addTriangle(triangle_indices[0], triangle_indices[1], triangle_indices[2], f);
        m_faces.push_back(f);
    }
}

template<typename VertexT, typename NormalT>
void BilinearFastBox<VertexT, NormalT>::optimizePlanarFaces(typename PointsetSurface<VertexT>::Ptr surface)
{
    // Check MC case
    if(this->m_mcIndex == 51 || this->m_mcIndex == 204)
    {
        typename SearchTree<VertexT>::Ptr tree = surface->searchTree();

        HalfEdgeFace<VertexT, NormalT>* edgeFaces[4];
        for(int i = 0; i < m_faces.size(); i++)
        {
            // do something...
        }
    }
}

template<typename VertexT, typename NormalT>
BilinearFastBox<VertexT, NormalT>::~BilinearFastBox()
{
    // TODO Auto-generated destructor stub
}

} /* namespace lssr */
