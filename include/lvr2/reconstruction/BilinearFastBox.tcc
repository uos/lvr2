/**
 * Copyright (c) 2018, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
BilinearFastBox<BaseVecT>::BilinearFastBox(BaseVecT center)
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
     BaseVecT corners[8];
     BaseVecT vertex_positions[12];

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
     for(int a = 0; MCTable[index][a] != -1; a+= 3)
     {
         OptionalVertexHandle vertex_indices[3];

         for(int b = 0; b < 3; b++)
         {
             auto edge_index = MCTable[index][a + b];

             //If no index was found generate new index and vertex
             //and update all neighbor boxes
             if(!this->m_intersections[edge_index])
             {
                 auto p = vertex_positions[edge_index];
                 this->m_intersections[edge_index] = mesh.addVertex(p);

                 for(int i = 0; i < 3; i++)
                 {
                     auto current_neighbor = this->m_neighbors[neighbor_table[edge_index][i]];
                     if(current_neighbor != 0)
                     {
                         current_neighbor->m_intersections[neighbor_vertex_table[edge_index][i]] = this->m_intersections[edge_index];
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
 void BilinearFastBox<BaseVecT>::optimizePlanarFaces(BaseMesh<BaseVecT>& mesh, size_t kc)
 {
     if(this->m_surface)
     {
         auto tree = this->m_surface->searchTree();
         vector<EdgeHandle> out_edges;

         for(auto face_it : m_faces)
         {
            auto edges = mesh.getEdgesOfFace(face_it);
            for(auto edge_it : edges)
            {
                if(mesh.isBorderEdge(edge_it))
                {
                    out_edges.push_back(edge_it);
                }
            }
         }

        if(out_edges.size() == 1 || out_edges.size() == 2 )
        {
            for(size_t i = 0; i < out_edges.size(); i++)
            {
                vector<size_t> nearest1, nearest2;

                auto vertices = mesh.getVerticesOfEdge(out_edges[i]);
                BaseVecT& p1 = mesh.getVertexPosition(vertices[0]);
                BaseVecT& p2 = mesh.getVertexPosition(vertices[1]);

                this->m_surface->searchTree()->kSearch(p1, kc, nearest1);
                size_t nk = min(kc, nearest1.size());
                FloatChannel pts = *(m_surface->pointBuffer()->getFloatChannel("points"));

                //Hmmm, sometimes the k-search seems to fail...
                if(nk > 0)
                {
                    BaseVecT centroid1;
                    for(auto idx : nearest1)
                    {
                        
                        BaseVecT p = pts[idx];
                        centroid1 += p;
                    }
                    centroid1 /= nk;

                    p1[0] = centroid1[0];
                    p1[1] = centroid1[1];
                    p1[2] = centroid1[2];
                }

                this->m_surface->searchTree()->kSearch(p2, kc, nearest2);
                nk = min(kc, nearest2.size());

                //Hmmm, sometimes the k-search seems to fail...
                if(nk > 0)
                {
                    BaseVecT centroid2;
                    for(auto idx : nearest2)
                    {
                        BaseVecT p = pts[idx];
                        centroid2 += p;
                    }
                    centroid2 /= nk;

                    p2[0] = centroid2[0];
                    p2[1] = centroid2[1];
                    p2[2] = centroid2[2];
                }
            }
        }
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
