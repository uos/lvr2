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
 * MeshNavAlgorithms.tcc
 */


#include <lvr2/attrmaps/AttrMaps.hpp>

namespace lvr2
{

template <typename BaseVecT>
void calcVertexLocalNeighborhood(const BaseMesh<BaseVecT>& mesh, VertexHandle vH, double radius, vector<VertexHandle>& neighbors)
{
    vector<VertexHandle> stack;
    stack.push_back(vH);
    while(!stack.empty())
    {
        auto cur_vH = stack.back();
        stack.pop_back();
        SparseVertexMap<bool> used_vertices(false);
        used_vertices[cur_vH] = true;

        vector<EdgeHandle> cur_edges = mesh.getEdgesOfVertex(cur_vH);
        for (auto eH: cur_edges)
        {
            vector<VertexHandle> vertex_vector = mesh.getVerticesOfEdge(eH);
            if (!used_vertices[vertex_vector[0]]) // add distance check vertex_vector[0] -> vH < radius
            {
                stack.push_back(vertex_vector[0]);
                neighbors.push_back(vertex_vector[0]);
            }
            else
            {
                if (!used_vertices[vertex_vector[1]]) // same dist check as above
                {
                    stack.push_back(vertex_vector[1]);
                    neighbors.push_back(vertex_vector[1]);
                }
            };
        }
    }
}

template <typename BaseVecT>
DenseVertexMap<float> calcVertexHeightDiff(const BaseMesh<BaseVecT>& mesh, double radius)
{
    DenseVertexMap<float> height_diff;

    // calculate height difference for each vertex
    for (auto vH: mesh.vertices())
    {
        // get neighbored vertices
        vector<VertexHandle> neighbors;
        calcVertexLocalNeighborhood(mesh, vH, radius, neighbors);

        // store initial values for min and max height
        double min_height = std::numeric_limits<float>::max();
        double max_height = std::numeric_limits<float>::min();

        // adjust the min and max height values, according to the neighborhood
        typename vector<VertexHandle>::iterator neighbor_iter;
        for (neighbor_iter = neighbors.begin(); neighbor_iter != neighbors.end(); neighbor_iter++)
        {
            auto cur_neighbor = (*neighbor_iter);
            auto cur_pos = mesh.getVertexPosition(cur_neighbor);
            min_height = std::min(cur_pos.z, min_height);
            max_height = std::max(cur_pos.z, max_height);
        }

        // calculate the final height difference
        height_diff[vH] = max_height - min_height;
        return height_diff;
    }
}

} // namespace lvr2
