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
 * TetraederBox.cpp
 *
 *  @date 23.11.2011
 *  @author Thomas Wiemann
 */

#include "TetraederTables.hpp"

namespace lvr2
{

template<typename BaseVecT>
TetraederBox<BaseVecT>::TetraederBox(Vector<BaseVecT> v) : FastBox<BaseVecT>(v)
{
    //for(int i = 0; i < 19; i++) this->m_intersections[i] = this->INVALID_INDEX;
}

template<typename BaseVecT>
void TetraederBox<BaseVecT>::interpolateIntersections(
        int tetraNumber,
        Vector<BaseVecT> positions[4],
        float distances[4]
        )
{
    // Index variables
    int v1;
    int v2;
    float x;
    float y;
    float z;

    // Calc intersections for tetraeders
    v1 = 0;
    v2 = 1;
    x = this->calcIntersection(positions[v1].x, positions[v2].x, distances[v1], distances[v2]);
    y = this->calcIntersection(positions[v1].y, positions[v2].y, distances[v1], distances[v2]);
    z = this->calcIntersection(positions[v1].z, positions[v2].z, distances[v1], distances[v2]);
    m_intersectionPositionsTetraeder[0] = Vector<BaseVecT>(x, y, z);

    v1 = 1;
    v2 = 3;
    x = this->calcIntersection(positions[v1].x, positions[v2].x, distances[v1], distances[v2]);
    y = this->calcIntersection(positions[v1].y, positions[v2].y, distances[v1], distances[v2]);
    z = this->calcIntersection(positions[v1].z, positions[v2].z, distances[v1], distances[v2]);
    m_intersectionPositionsTetraeder[1] = Vector<BaseVecT>(x, y, z);

    v1 = 3;
    v2 = 0;
    x = this->calcIntersection(positions[v1].x, positions[v2].x, distances[v1], distances[v2]);
    y = this->calcIntersection(positions[v1].y, positions[v2].y, distances[v1], distances[v2]);
    z = this->calcIntersection(positions[v1].z, positions[v2].z, distances[v1], distances[v2]);
    m_intersectionPositionsTetraeder[2] = Vector<BaseVecT>(x, y, z);

    v1 = 0;
    v2 = 2;
    x = this->calcIntersection(positions[v1].x, positions[v2].x, distances[v1], distances[v2]);
    y = this->calcIntersection(positions[v1].y, positions[v2].y, distances[v1], distances[v2]);
    z = this->calcIntersection(positions[v1].z, positions[v2].z, distances[v1], distances[v2]);
    m_intersectionPositionsTetraeder[3] = Vector<BaseVecT>(x, y, z);

    v1 = 1;
    v2 = 2;
    x = this->calcIntersection(positions[v1].x, positions[v2].x, distances[v1], distances[v2]);
    y = this->calcIntersection(positions[v1].y, positions[v2].y, distances[v1], distances[v2]);
    z = this->calcIntersection(positions[v1].z, positions[v2].z, distances[v1], distances[v2]);
    m_intersectionPositionsTetraeder[4] = Vector<BaseVecT>(x, y, z);

    v1 = 3;
    v2 = 2;
    x = this->calcIntersection(positions[v1].x, positions[v2].x, distances[v1], distances[v2]);
    y = this->calcIntersection(positions[v1].y, positions[v2].y, distances[v1], distances[v2]);
    z = this->calcIntersection(positions[v1].z, positions[v2].z, distances[v1], distances[v2]);
    m_intersectionPositionsTetraeder[5] = Vector<BaseVecT>(x, y, z);

}

template<typename BaseVecT>
TetraederBox<BaseVecT>::~TetraederBox()
{

}

template<typename BaseVecT>
void TetraederBox<BaseVecT>::getSurface(
        BaseMesh<BaseVecT> &mesh,
        vector<QueryPoint<BaseVecT> > &query_points,
        uint &globalIndex)
{
    typedef TetraederBox<BaseVecT>*  p_tBox;

    // Calc the vertex positions for all possible edge intersection
    // of all tetraeders (up to 19 for a single box)
    //Vector<BaseVecT> intersection_positions[19];

    // Sub-divide the box into six tetraeders using the existing
    // box corners. The defintions of the six tetraeders can be
    // found in the TetraederDefinitionTable.
    for(int t_number = 0; t_number < 6; t_number++)
    {
        //cout << "NEW TETRA" << " " << t_number <<  endl;
        // Get the 4 vertices of the current tetraeder
        Vector<BaseVecT> t_vertices[4];
        for(int i = 0; i < 4; i++)
        {
            t_vertices[i] =
                    query_points[this->m_vertices[TetraederDefinitionTable[t_number][i]]].m_position;
        }

        // Get the distance values for the four tetraeder
        // vertices
        float distances[4];
        for(int i = 0; i < 4; i++)
        {
            distances[i] =
                    query_points[this->m_vertices[TetraederDefinitionTable[t_number][i]]].m_distance;
        }

        // Interpolate the intersection vertices
        this->interpolateIntersections(t_number, t_vertices, distances);

        // Calculate the index for the surface generation look
        // up table
        int index = calcPatternIndex(distances);

        // Create the surface triangles
        OptionalVertexHandle triangle_indices[3];
        for(int a = 0; TetraederTable[index][a] != -1; a+= 3)
        {
            for(int b = 0; b < 3; b++)
            {
                // Map the 19 possible intersection points within this
                // box to a consistent numbering within the teraeder
                // so that we can use a common creation table for all
                // tetraeders.
                int edge_index = TetraederIntersectionTable[t_number][TetraederTable[index][a + b]];

                //If no index was found generate new index and vertex
                //and update all neighbor boxes
                if(!m_intersections[edge_index])
                {
                    Vector<BaseVecT> v = this->m_intersectionPositionsTetraeder[TetraederTable[index][a + b]];
                    OptionalVertexHandle handle =  mesh.addVertex(v);
                    this->m_intersections[edge_index] = handle;


                    //Update adjacent cells (three at most)
                    for(int i = 0; i < 3; i++)
                    {

                        int nb_index = TetraederNeighborTable[edge_index][i];

                        // Check if neighbor exists. The table contains
                        // a -1 flag, we can stop searching due to the
                        // structure of the tables
                        if(nb_index == -1)
                        {
                            break;
                        }

                        // Cast to correct correct type, we need a TetraederBox
                        p_tBox b = static_cast<p_tBox>(this->m_neighbors[nb_index]);

                        // Update index
                        if(b)
                        {
                            b->m_intersections[TetraederVertexNBTable[edge_index][i]] = handle;
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
            mesh.addFace(triangle_indices[0].unwrap(),
                         triangle_indices[1].unwrap(),
                         triangle_indices[2].unwrap());
        }
    }
}


} /* namespace lvr */
