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
void calcVertexLocalNeighborhood(
        const BaseMesh<BaseVecT>& mesh,
        VertexHandle vH,
        double radius,
        vector<VertexHandle>& neighborsOut)
{
    vector<VertexHandle> stack;
    stack.push_back(vH);
    SparseVertexMap<bool> usedVertices(false);

    while (!stack.empty())
    {
        auto curVH = stack.back();
        stack.pop_back();
        usedVertices.insert(curVH, true);

        vector<EdgeHandle> cur_edges = mesh.getEdgesOfVertex(curVH);
        for (auto eH: cur_edges)
        {
            auto vertexVector = mesh.getVerticesOfEdge(eH);

            if (!usedVertices[vertexVector[0]] && \
                 mesh.getVertexPosition(vertexVector[0]).distanceFrom(mesh.getVertexPosition(vH)) < radius)
            {
                stack.push_back(vertexVector[0]);
                neighborsOut.push_back(vertexVector[0]);
            }
            else
            {
                if (!usedVertices[vertexVector[1]] && \
                     mesh.getVertexPosition(vertexVector[1]).distanceFrom(mesh.getVertexPosition(vH)) < radius)
                {
                    stack.push_back(vertexVector[1]);
                    neighborsOut.push_back(vertexVector[1]);
                }
            }
        }
    }
}

template <typename BaseVecT>
DenseVertexMap<float> calcVertexHeightDiff(const BaseMesh<BaseVecT>& mesh, double radius)
{
    DenseVertexMap<float> height_diff;
    // Get neighbored vertices
    vector<VertexHandle> neighbors;

    // Calculate height difference for each vertex
    for (auto vH: mesh.vertices())
    {
        neighbors.clear();
        calcVertexLocalNeighborhood(mesh, vH, radius, neighbors);

        // Store initial values for min and max height
        float minHeight = std::numeric_limits<float>::max();
        float maxHeight = -std::numeric_limits<float>::max();

        // Adjust the min and max height values, according to the neighborhood
        for (auto neighbor: neighbors)
        {
            auto cur_pos = mesh.getVertexPosition(neighbor);
            minHeight = std::min(cur_pos.y, minHeight);
            maxHeight = std::max(cur_pos.y, maxHeight);
        }

        // Calculate the final height difference
        height_diff.insert(vH, maxHeight-minHeight);
    }

    return height_diff;
}

template<typename BaseVecT>
DenseEdgeMap<float> calcVertexAngleEdges(const BaseMesh<BaseVecT>& mesh, const VertexMap<Normal<BaseVecT>>& normals)
{
    DenseEdgeMap<float> edge_angle;

    for (auto eH: mesh.edges())
    {
        auto vH_vector = mesh.getVerticesOfEdge(eH);
        edge_angle.insert(eH, acos(normals[vH_vector[0]].dot(normals[vH_vector[1]].asVector())));
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
        auto edgeVec = mesh.getEdgesOfVertex(vH);
        int degree = edgeVec.size();
        for(auto eH: edgeVec)
        {
            angle_sum += edge_angles[eH];
        }
        vertex_angles.insert(vH, angle_sum/degree);
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
    auto average_angles = calcAverageVertexAngles(mesh, normals);

    // calculate roughness for each vertex
    for (auto vH: mesh.vertices())
    {
        sum = 0.0;

        neighbors.clear();
        calcVertexLocalNeighborhood(mesh, vH, radius, neighbors);


        // adjust sum values, according to the neighborhood
        for (auto neighbor: neighbors)
        {
           sum += average_angles[neighbor];
        }

        // calculate the final roughness
        roughness.insert(vH, sum / neighbors.size());

    }
    return roughness;

}

template<typename in, typename out, typename MapF>
DenseVertexMap<out> changeMap(const VertexMap<in>& map_in, MapF map_function)
{
    DenseVertexMap<out> resultMap;

    for (auto vH: map_in)
    {
        resultMap.insert(vH, map_function(map_in[vH]));
    }

    return resultMap;
}

} // namespace lvr2
