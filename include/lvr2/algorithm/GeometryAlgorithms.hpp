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

/*
 * GeometryAlgorithms.hpp
 */

#ifndef LVR2_ALGORITHM_GEOMETRYALGORITHMS_H_
#define LVR2_ALGORITHM_GEOMETRYALGORITHMS_H_

#include "lvr2/geometry/BaseMesh.hpp"
#include "lvr2/attrmaps/AttrMaps.hpp"
#include "lvr2/geometry/Handles.hpp"
#include <list>

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
 * @param mesh          The given BaseMesh for performing the neighborhood-search.
 * @param vH            The given VertexHandle to which we want to get the local heighborhood.
 * @param radius        The radius which defines the border of the local neighborhood.
 * @param neighborsOut  The found neighbors, stored in a vector.
 */
template<typename BaseVecT>
void calcLocalVertexNeighborhood(
        const BaseMesh<BaseVecT>& mesh,
        VertexHandle vH,
        double radius,
        vector<VertexHandle>& neighborsOut
);


/**
 * @brief Visits every vertex in the local neighborhood of `vH`.
 *
 * The local neighborhood is defined as all vertices that are connected to `vH`
 * and where the "path" in between those vertices only contains vertices that
 * are no further away from `vH` than `radius`.
 *
 * For every such vertex in the local neighborhood (not `vH` itself!) the
 * given `visitor` is called exactly once.
 */
template <typename BaseVecT, typename VisitorF>
void visitLocalVertexNeighborhood(
    const BaseMesh<BaseVecT>& mesh,
    VertexHandle vH,
    double radius,
    VisitorF visitor
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
DenseVertexMap<float> calcVertexHeightDifferences(const BaseMesh<BaseVecT>& mesh, double radius);

/**
 * @brief Calculates the roughness for each vertex.
 *
 * For the calculation of the roughness we sum up the average angles of neighbored vertices and divide that
 * by the size of the neighborhood.
 *
 * @param mesh      The given mesh for the calculation.
 * @param radius    The radius which defines the local neighborhood.
 *                  A local neighborhood is defined by a circular-shaped radius, which includes all connected
 *                  edges and vertices. Once an edge leaves this radius and reenters somehow it isn't part of
 *                  the neighborhood.
 * @param normals   The vertex normals of the given mesh as a map.
 *                  The normals are necessary in this function for delegating them to the submethods.
 *
 * @return A map <vertex, float> filled with roughness values for each vertex.
 */
template<typename BaseVecT>
DenseVertexMap<float> calcVertexRoughness(
        const BaseMesh<BaseVecT>& mesh,
        double radius,
        const VertexMap<Normal<typename BaseVecT::CoordType>>& normals
);

/**
 * @brief Calculates the average angle for each vertex.
 *
 * @param mesh      The given mesh for the calculation.
 * @param normals   The vertex normals of the given mesh.
 *                  The normals are necessary in this function for delegating them to the submethods.
 *
 * @return A map <vertex, float> with the average angle for each vertex.
 */
template<typename BaseVecT>
DenseVertexMap<float> calcAverageVertexAngles(
        const BaseMesh<BaseVecT>& mesh,
        const VertexMap<Normal<typename BaseVecT::CoordType>>& normals
);

/**
 * @brief Calculates the angle between two vertex normals for each edge.
 *
 * @param mesh      The given mesh for the calculation.
 * @param normals   The vertex normals of the given mesh.
 *                  The normals are necessary in this function for delegating them to the submethods.
 *
 * @return A map <edge, float> with the angle of each edge.
 */
template<typename BaseVecT>
DenseEdgeMap<float> calcVertexAngleEdges(
        const BaseMesh<BaseVecT>& mesh,
        const VertexMap<Normal<typename BaseVecT::CoordType>>& normals
);

/**
 * @brief Calculates the roughness and the height difference for each vertex.
 *
 * This function combines the logic of the calcVertexRoughness- and calcVertexHeightDiff-functions,
 * allowing us to calculate the local neighborhood for each single vertex just once.
 * By that, this function should always be used when the roughness and height difference values are
 * both needed.
 *
 * @param mesh        The given mesh for the calculation.
 * @param radius      The radius which defines the local neighborhood.
 *                    A local neighborhood is defined by a circular-shaped radius, which includes all connected
 *                    edges and vertices. Once an edge leaves this radius and reenters somehow, it isn't part of
 *                    the neighborhood.
 * @param normals     The vertex normals of the given mesh.
 * @param roughness   The calculated roughness values for each vertex.
 * @param heightDiff  The calculated height difference values for each vertex.
 */
template<typename BaseVecT>
void calcVertexRoughnessAndHeightDifferences(
        const BaseMesh<BaseVecT>& mesh,
        double radius,
        const VertexMap<Normal<typename BaseVecT::CoordType>>& normals,
        DenseVertexMap<float>& roughness,
        DenseVertexMap<float>& heightDiff
);

/**
 * @brief Computes the distances between the vertices and stores them in the given dense edge map.
 *
 * @param mesh        The mesh containing the vertices and edges of interest
 * @return            The dense edge map with the distance values
 */
template<typename BaseVecT>
DenseEdgeMap<float> calcVertexDistances(const BaseMesh<BaseVecT>& mesh);

/**
 * @brief  Dijkstra's algorithm
 *
 * @param mesh        The mesh containing the vertices and edges of interest
 * @param start       Start vertex
 * @param goal        Goal vertex
 * @param path        Resulted path from search
 *
 * @return true if a path between start and goals exists
 */
template<typename BaseVecT>
bool Dijkstra(
    const BaseMesh<BaseVecT>& mesh,
    const VertexHandle& start,
    const VertexHandle& goal,
    const DenseEdgeMap<float>& edgeCosts,
    std::list<VertexHandle>& path,
    DenseVertexMap<float>& distances,
    DenseVertexMap<VertexHandle>& predecessors,
    DenseVertexMap<bool>& seen,
    DenseVertexMap<float>& vertex_costs);


} // namespace lvr2

#include "GeometryAlgorithms.tcc"

#endif /* LVR2_ALGORITHM_GEOMETRYALGORITHMS_H_ */
