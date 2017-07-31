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

#include <algorithm>
#include <limits>

#include <lvr2/attrmaps/AttrMaps.hpp>


namespace lvr2
{

template <typename BaseVecT>
void calcVertexLocalNeighborhood(const BaseMesh<BaseVecT>& mesh, VertexHandle vH, double radius, vector<VertexHandle>& neighbors)
{
    vector<VertexHandle> stack;
    stack.push_back(vH);
    SparseVertexMap<bool> used_vertices(false);

    while(!stack.empty())
    {
        auto cur_vH = stack.back();
        stack.pop_back();
        used_vertices.insert(cur_vH, true);

        vector<EdgeHandle> cur_edges = mesh.getEdgesOfVertex(cur_vH);
        for (auto eH: cur_edges)
        {
            auto vertex_vector = mesh.getVerticesOfEdge(eH);
            //cout << "Current Vertex: " << mesh.getVertexPosition(vH).x << ", " << mesh.getVertexPosition(vH).y << ", " << mesh.getVertexPosition(vH).z << endl;
            //cout << "Vertex 1: " << mesh.getVertexPosition(vertex_vector[0]).x << ", " << mesh.getVertexPosition(vertex_vector[0]).y << ", " << mesh.getVertexPosition(vertex_vector[0]).z << endl;
            //cout << "Current Distance: " << mesh.getVertexPosition(vertex_vector[0]).distanceFrom(mesh.getVertexPosition(vH)) << endl;
            if (!used_vertices[vertex_vector[0]] && \
                 mesh.getVertexPosition(vertex_vector[0]).distanceFrom(mesh.getVertexPosition(vH)) < radius)
            {
                stack.push_back(vertex_vector[0]);
                neighbors.push_back(vertex_vector[0]);
                //cout << "if 1" << endl;
            }
            else
            {
                if (!used_vertices[vertex_vector[1]] && \
                     mesh.getVertexPosition(vertex_vector[1]).distanceFrom(mesh.getVertexPosition(vH)) < radius)
                {
                    stack.push_back(vertex_vector[1]);
                    neighbors.push_back(vertex_vector[1]);
                    //cout << "if 2" << endl;
                }
            }
        }
        //cout << "current neighbor size: " << neighbors.size() << endl;
    }
}

template <typename BaseVecT>
DenseVertexMap<float> calcVertexHeightDiff(const BaseMesh<BaseVecT>& mesh, double radius)
{
    DenseVertexMap<float> height_diff;
    // get neighbored vertices
    vector<VertexHandle> neighbors;

    // calculate height difference for each vertex
    for (auto vH: mesh.vertices())
    {
        neighbors.clear();
        //cout << "Neighborvector size 1: " << neighbors.size() << endl;
        calcVertexLocalNeighborhood(mesh, vH, radius, neighbors);
        //cout << "Neighborvector size 2: " << neighbors.size() << endl;

        // store initial values for min and max height
        float min_height = std::numeric_limits<float>::max();
        float max_height = std::numeric_limits<float>::min();

        // adjust the min and max height values, according to the neighborhood
        for (auto neighbor: neighbors)
        {
            auto cur_pos = mesh.getVertexPosition(neighbor);
            //cout << "Current Position (y-value): " << cur_pos.y << endl;
            min_height = std::min(cur_pos.y, min_height);
            max_height = std::max(cur_pos.y, max_height);
        }

        cout << "Max Height:" << max_height << endl;
        cout << "Min Height:" << min_height << endl;

        // calculate the final height difference
        height_diff.insert(vH, max_height-min_height);
    }

    return height_diff;
}

template<typename in, typename out, typename MapF>
DenseVertexMap<out> changeMap(const VertexMap<in>& map_in, MapF map_function)
{
    DenseVertexMap<out> result_map;

    for (auto vH: map_in)
    {
        result_map.insert(vH, map_function(map_in[vH]));
    }

    return result_map;
}

} // namespace lvr2
