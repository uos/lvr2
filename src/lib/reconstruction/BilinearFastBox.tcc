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

static int PlaneTable[28] =
{
        204, 51, 153, 102, 240, 15,
        192, 63, 48, 207, 12, 243, 3, 252, 96, 144, 6, 9, 159, 111, 249, 246, 136, 119, 17, 238, 34, 221
};

template<typename VertexT, typename NormalT>
BilinearFastBox<VertexT, NormalT>::BilinearFastBox(VertexT &center)
    : FastBox<VertexT, NormalT>(center)
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
void BilinearFastBox<VertexT, NormalT>::optimizePlanarFaces(typename PointsetSurface<VertexT>::Ptr surface)
{
    typedef HalfEdge<HalfEdgeVertex<VertexT, NormalT>, HalfEdgeFace<VertexT, NormalT> > HEdge;

    // Check MC case
    for(int a = 0; a < 28; a++)
    {

        if(this->m_mcIndex == PlaneTable[a])
        {
            typename SearchTree<VertexT>::Ptr tree = surface->searchTree();

            // Detect triangles that are on the border of the mesh
            vector<HEdge*> out_edges;
            for(int i = 0; i < m_faces.size(); i++)
            {
                HalfEdgeFace<VertexT, NormalT>* face = m_faces[i];

                HEdge* e = face->m_edge;
                HEdge* f = face->m_edge->next;
                HEdge* g = face->m_edge->next->next;
                if(e->pair->face == 0) out_edges.push_back(e);
                if(f->pair->face == 0) out_edges.push_back(f);
                if(g->pair->face == 0) out_edges.push_back(g);

            }

            // Handle different cases
            if(out_edges.size() == 1 || out_edges.size() == 2 )
            {
                // Get nearest points
                for(int i = 0; i < out_edges.size(); i++)
                {

                    vector<VertexT> nearest1, nearest2;
                    tree->kSearch( out_edges[i]->start->m_position, 1, nearest1);

                    // Hmmm, sometimes the k-search seems to fail...
                    if(nearest1.size() > 0)
                    {
                        out_edges[i]->start->m_position = nearest1[0];
                    }

                    tree->kSearch( out_edges[i]->end->m_position, 1, nearest2);

                    if(nearest2.size() > 0)
                    {
                        out_edges[i]->end->m_position = nearest2[0];
                    }

                }
            }
        }
    }
}

template<typename VertexT, typename NormalT>
BilinearFastBox<VertexT, NormalT>::~BilinearFastBox()
{
    // TODO Auto-generated destructor stub
}

} /* namespace lssr */
