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

namespace lvr2
{

template<typename BaseVecT>
PointsetSurfacePtr<BaseVecT> BilinearFastBox<BaseVecT>::m_surface = 0;


template<typename BaseVecT>
const string BoxTraits<BilinearFastBox<BaseVecT>>::type = "BilinearFastBox";


template<typename BaseVecT>
BilinearFastBox<BaseVecT>::BilinearFastBox(Point<BaseVecT> center)
    : FastBox<BaseVecT>(center), m_mcIndex(0)
{
    //cout << m_surface << endl;
}

template<typename BaseVecT>
void BilinearFastBox<BaseVecT>::getSurface(
        BaseMesh<BaseVecT>& mesh,
        vector<QueryPoint<BaseVecT>> &qp,
        uint &globalIndex)
{
    //FastBox<BaseVecT>::getSurface(mesh, qp, globalIndex);
     Point<BaseVecT> corners[8];
     Point<BaseVecT> vertex_positions[12];

     float distances[8];

     this->getCorners(corners, qp);
     this->getDistances(distances, qp);
     this->getIntersections(corners, distances, vertex_positions);

     int index = this->getIndex(qp);

     // Do not create triangles for invalid boxes
     for (int i = 0; i < 8; i++)
     {
         if (qp[this->m_vertices[i]].m_invalid)
         {
             return;
         }
     }

     // Generate the local approximation surface according to the marching
     // cubes table for Paul Burke.
     for(int a = 0; lvr::MCTable[index][a] != -1; a+= 3)
     {
         OptionalVertexHandle vertex_indices[3];

         for(int b = 0; b < 3; b++)
         {
             auto edge_index = lvr::MCTable[index][a + b];

             //If no index was found generate new index and vertex
             //and update all neighbor boxes
             if(!this->m_intersections[edge_index])
             {
                 auto p = vertex_positions[edge_index];
                 this->m_intersections[edge_index] = mesh.addVertex(p);

                 for(int i = 0; i < 3; i++)
                 {
                     auto current_neighbor = this->m_neighbors[lvr::neighbor_table[edge_index][i]];
                     if(current_neighbor != 0)
                     {
                         current_neighbor->m_intersections[lvr::neighbor_vertex_table[edge_index][i]] = this->m_intersections[edge_index];
                     }
                 }

                 // Increase the global vertex counter to save the buffer
                 // position were the next new vertex has to be inserted
                 globalIndex++;
             }

             //Save vertex index in mesh
             vertex_indices[b] = this->m_intersections[edge_index];
         }

         // Add triangle actually does the normal interpolation for us.
         auto f = mesh.addFace(
             vertex_indices[0].unwrap(),
             vertex_indices[1].unwrap(),
             vertex_indices[2].unwrap()
         );
         m_faces.push_back(f); // THIS IS THE ONLY LINE DIFFERENT FROM BASE IMPL
     }
}

 template<typename BaseVecT>
 void BilinearFastBox<BaseVecT>::optimizePlanarFaces(size_t kc)
 {
     if(this->m_surface)
     {

//         auto tree = this->m_surface->searchTree();

//         // Detect triangles that are on the border of the mesh
//         vector<EdgeHandle> out_edges;

//         for(int i = 0; i < m_faces.size(); i++)
//         {
//             auto face = m_faces[i];
//             HEdge* e = face->m_edge;
//             for(int j = 0; j < 2; j++)
//             {
//                 // Catch null pointer from outer faces
//                 try
//                 {
//                     e->pair()->face();
//                 }
//                 catch (HalfEdgeAccessException& ex)
//                 {
//                     out_edges.push_back(e);
//                 }

//                 // Check integrity
//                 try
//                 {
//                     e = e->next();
//                 }
//                 catch (HalfEdgeAccessException& ex)
//                 {
//                     // Face corrupted, abort
//                     cout << "Warning, corrupted face" << endl;
//                     break;
//                 }
//             }

//         }

//         // Handle different cases
//         if(out_edges.size() == 1 || out_edges.size() == 2 )
//         {
//             // Get nearest points
//             for(int i = 0; i < out_edges.size(); i++)
//             {
//                 vector<VertexT> nearest1, nearest2;
//                 this->m_surface->searchTree()->kSearch( out_edges[i]->start()->m_position, kc, nearest1);

//                 size_t nk = min(kc, nearest1.size());


//                 // Hmmm, sometimes the k-search seems to fail...
//                 if(nk > 0)
//                 {
//                     VertexT centroid1;
//                     for(int a = 0; a < nk; a++)
//                     {
//                         centroid1 += nearest1[a];
//                     }
//                     centroid1 /= nk;
//                     out_edges[i]->start()->m_position = centroid1;
//                 }

//                 this->m_surface->searchTree()->kSearch( out_edges[i]->end()->m_position, kc, nearest2);
//                 nk = min(kc, nearest2.size());

//                 if(nk > 0)
//                 {
//                     VertexT centroid2;
//                     for(int a = 0; a < nk; a++)
//                     {
//                         centroid2 += nearest2[a];
//                     }
//                     centroid2 /= nk;
//                     out_edges[i]->end()->m_position = centroid2;
//                 }

//             }
//         }
     }
 }

template<typename BaseVecT>
BilinearFastBox<BaseVecT>::~BilinearFastBox()
{}

template<typename BaseVecT>
void BilinearFastBox<BaseVecT>::getSurface(
    BaseMesh<BaseVecT>& mesh,
    vector<QueryPoint<BaseVecT>>& query_points,
    uint& globalIndex,
    BoundingBox<BaseVecT>& bb,
    vector<unsigned int>& duplicates,
    float comparePrecision
)
{
    FastBox<BaseVecT>::getSurface(mesh, query_points, globalIndex, bb, duplicates, comparePrecision);
}

} // namespace lvr2
