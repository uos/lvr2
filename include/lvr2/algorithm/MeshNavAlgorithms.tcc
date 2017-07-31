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
    while(!stack.empty())
    {
        auto cur_vH = stack.back();
        stack.pop_back();
        SparseVertexMap<bool> used_vertices(false);
        used_vertices[cur_vH] = true;

        vector<EdgeHandle> cur_edges = mesh.getEdgesOfVertex(cur_vH);
        for (auto eH: cur_edges)
        {
            auto vertex_vector = mesh.getVerticesOfEdge(eH);
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
    // get neighbored vertices
    vector<VertexHandle> neighbors;

    // calculate height difference for each vertex
    for (auto vH: mesh.vertices())
    {
        neighbors.clear();
        calcVertexLocalNeighborhood(mesh, vH, radius, neighbors);

        // store initial values for min and max height
        double min_height = std::numeric_limits<double>::max();
        double max_height = std::numeric_limits<double>::min();

        // adjust the min and max height values, according to the neighborhood
        for (auto neighbor: neighbors)
        {
            auto cur_neighbor = neighbor;
            auto cur_pos = mesh.getVertexPosition(cur_neighbor);
            min_height = std::min(cur_pos.z, min_height);
            max_height = std::max(cur_pos.z, max_height);
        }

        // calculate the final height difference
        height_diff[vH] = max_height - min_height;
        
        return height_diff;
    }
}

template<typename BaseVecT>
DenseEdgeMap<float> calcVertexAngleEdges(const BaseMesh<BaseVecT>& mesh, const VertexMap<Normal<BaseVecT>>& normals)
{
    DenseEdgeMap<float> edge_angle;

    for (auto eH: mesh.edges())
    {
        auto vH_vector = mesh.getVerticesOfEdge(eH);
        edge_angle.insert(eH, acos(normals[vH_vector[0]]*normals[vH_vector[1]]));
        if(isnan(edge_angle[eH]))
        {
                edge_angle[eH] = 0;
        }
    }
    return edge_angle;
}

template<typename BaseVecT>
DenseVertexMap<float> calcAverageVertexAngles(const BaseMesh<BaseVecT>& mesh, const VertexMap<Normal<BaseVecT>>& normals)
{
    DenseVertexMap<float> vertex_angles;
    auto edge_angles = calcVertexAngleEdges(mesh, normals);
    float angle_sum = 0;

    for (auto vH: mesh.vertices())
    {
        angle_sum = 0;
        auto edgeVec = mesh.getEdgesofVertex(vH);
        int degree = edgeVec.size();
        for(auto eH: edgeVec)
        {
            angle_sum += edge_angles[eH];
        }
        vertex_angles[vH] = angle_sum/degree;
    }

    return vertex_angles;
}


    template<typename BaseVecT>
DenseVertexMap<float> calcVertexRoughness(const BaseMesh<BaseVecT>& mesh, double radius, const VertexMap<Normal<BaseVecT>>& normals)
{
    DenseVertexMap<float> roughness;
    // get neighbored vertices
    vector<VertexHandle> neighbors;
    double sum;


    // calculate roughness for each vertex
    for (auto vH: mesh.vertices())
    {
        auto average_angles = calcAverageVertexAngles(mesh, normals);
        sum = 0.0;

        neighbors.clear();
        calcVertexLocalNeighborhood(mesh, vH, radius, neighbors);


        // adjust sum values, according to the neighborhood
        for (auto neighbor: neighbors)
        {
            auto cur_neighbor = neighbor;

        }

        // calculate the final roughness
        roughness[vH] = sum / neighbors.size();

        }
    return roughness;

}

template <typename MapF, typename in, typename out>
DenseVertexMap<out> map(const VertexMap<in>& map_in, MapF map_function)
{
    DenseVertexMap<out> result_map;

    for (auto vH: map_in)
    {
        result_map[vH] = map_function(map_in[vH]);
    }

    return result_map;
}

} // namespace lvr2
