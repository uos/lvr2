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
 * MeshNavAlgorithms.hpp
 */

#ifndef LVR2_ALGORITHM_MESHNAVALGORITHMS_H_
#define LVR2_ALGORITHM_MESHNAVALGORITHMS_H_

#include <lvr2/geometry/BaseMesh.hpp>
#include <lvr2/attrmaps/AttrMaps.hpp>
#include <lvr2/geometry/Handles.hpp>

namespace lvr2
{

/**
 * @brief   Calculates the local neighborhood of a given vertex (defined by it's handle).
 *
 * A local neighborhood for a vertex is constrained by a circular-shaped radius.
 * The neighbors of a vertex do have to be connected by vertices and edges, which stay within this border. If one
 * edge leaves the neighborhood radius, every further on connected vertex and edge isn't part of the local
 * neighborhood, even if the topological "edge->vertex->edge->vertex->..."-chain, which left the radius once,
 * reenters the radius.
 *
 * @param mesh      The given BaseMesh for performing the neighborhood-search.
 * @param vH        The given VertexHandle to which we want to get the local heighborhood.
 * @param radius    The radius which defines the border of the local neighborhood.
 * @param neighbors The found neighbors, stored in a vector.
 */
template<typename BaseVecT>
void calcVertexLocalNeighborhood(
        const BaseMesh<BaseVecT>& mesh,
        VertexHandle vH,
        double radius,
        vector<VertexHandle>& neighborsOut
);

/**
 * @brief   Calculate the height difference value for each vertex of the given BaseMesh.
 *
 * @param mesh      The given BaseMesh for calculating vertex height differences.
 * @param radius    The radius which defines the border of the local neighborhood.
 *
 * @return  A map filled with <Vertex, float>-entries, storing the height difference value
 *          of each vertex.
 */
template<typename BaseVecT>
DenseVertexMap<float> calcVertexHeightDiff(const BaseMesh<BaseVecT>& mesh, double radius);

/**
 * @brief calculates the roughness for each vertex
 *
 * @param mesh given mesh for calculation
 * @param radius radius whish defines the local neighborhood
 * @param normals the vertex norals of the given mesh as a map
 * @return A map <vertex, float> filled with roughness values for each vertex
 */
template<typename BaseVecT>
DenseVertexMap<float> calcVertexRoughness(const BaseMesh<BaseVecT>& mesh, double radius, const VertexMap<Normal<BaseVecT>>& normals);

/**
 * @brief calculates the average angle for each vertex
 *
 * @param mesh given mesh for calculation
 * @param normals vertex normals of the mesh
 * @return map<vertex, float> with the average angle for each vertex
 */
template<typename BaseVecT>
DenseVertexMap<float> calcAverageVertexAngles(const BaseMesh<BaseVecT>& mesh, const VertexMap<Normal<BaseVecT>>& normals);

/**
 * @brief calculates the angle for each edge
 *
 * @param mesh given mesh for calculation
 * @param normals vertex normals of the mesh
 * @return map<edge, float> with the angle of each edge
 */
template<typename BaseVecT>
DenseEdgeMap<float> calcVertexAngleEdges(const BaseMesh<BaseVecT>& mesh, const VertexMap<Normal<BaseVecT>>& normals);

/**
 * @brief   Change the given input VertexMap to a different type output DenseVertexMap.
 *
 * The conversion between the input VertexMap and output DenseVertexMap is defined by the given function.
 * The method changeMap takes every element of the input VertexMap and converts it individually via the given
 * map-conversion function and adds the converted element to the output DenseVertexMap.
 *
 * @tparam  in              Templatetype for the input VertexMap.
 * @tparam  out             Templatetype for the output DenseVertexMap.
 * @tparam  MapF            Templatetype for the given map-conversion function.
 * @param   map_in          VertexMap holding input-type data, which will be
 *                          converted to the DenseVertexMap holding output-type data.
 * @param   map_function    Function for converting a single element of the
 *                          input-type to a single element of the output-type.
 *
 * @return  A DenseVertexMap holding output-type data,
 *          which is created via the input VertexMap and the map-conversion function.
 */
template<typename in, typename out, typename MapF>
DenseVertexMap<out> changeMap(const VertexMap<in>& map_in, MapF map_function);

} // namespace lvr2

#include <lvr2/algorithm/MeshNavAlgorithms.tcc>

#endif /* LVR2_ALGORITHM_MESHNAVALGORITHMS_H_ */
