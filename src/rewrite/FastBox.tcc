/*
 * FastBox.cpp
 *
 *  Created on: 03.03.2011
 *      Author: Thomas Wiemann
 */

namespace lssr
{

template<typename CoordType, typename IndexType>
CoordType FastBox<CoordType, IndexType>::m_voxelsize = 0;

template<typename CoordType, typename IndexType>
FastBox<CoordType, IndexType>::FastBox(Vertex<CoordType> &center)
{
    // Init members
    for(int i = 0; i < 12; i++) m_intersections[i] = 0;
    m_center = center;
}

template<typename CoordType, typename IndexType>
void FastBox<CoordType, IndexType>::setVertex(int index, IndexType nb)
{
    m_vertices[index] = nb;
}

template<typename CoordType, typename IndexType>
void FastBox<CoordType, IndexType>::setNeighbor(int index, FastBox<CoordType, IndexType>* nb)
{
    m_neighbors[index] = nb;
}


template<typename CoordType, typename IndexType>
FastBox<CoordType, IndexType>* FastBox<CoordType, IndexType>::getNeighbor(int index)
{
    return m_neighbors[index];
}

template<typename CoordType, typename IndexType>
IndexType FastBox<CoordType, IndexType>::getVertex(int index)
{
    return m_vertices[index];
}



template<typename CoordType, typename IndexType>
void FastBox<CoordType, IndexType>::getCorners(Vertex<CoordType> corners[],
                                               vector<QueryPoint<CoordType> > &qp)
{
    // Get the box corner positions from the query point array
    for(int i = 0; i < 8; i++){
        corners[i] = Vertex<CoordType>(qp[m_vertices[i]].m_position);
    }
}

template<typename CoordType, typename IndexType>
void FastBox<CoordType, IndexType>::getDistances(CoordType distances[],
                                                 vector<QueryPoint<CoordType> > &qp)
{
    // Get the distance values from the query point array
    // for the corners of the current box
    for(int i = 0; i < 8; i++)
    {
        distances[i] = qp[m_vertices[i]].m_distance;
    }
}

template<typename CoordType, typename IndexType>
int  FastBox<CoordType, IndexType>::getIndex(vector<QueryPoint<CoordType> > &qp)
{
    // Determine the MC-Table index for the current corner configuration
    int index = 0;
    for(int i = 0; i < 8; i++)
    {
        if(qp[m_vertices[i]].m_distance > 0) index |= (1 << i);
    }
    return index;
}

template<typename CoordType, typename IndexType>
CoordType FastBox<CoordType, IndexType>::calcIntersection(CoordType x1, CoordType x2, CoordType d1, CoordType d2)
{
    // Calculate the surface intersection using linear interpolation
    return  x2 - d2 * (x1 - x2) / (d1 - d2);
}

template<typename CoordType, typename IndexType>
void FastBox<CoordType, IndexType>::getIntersections(Vertex<CoordType> corners[],
                                                     CoordType distance[],
                                                     Vertex<CoordType> positions[])
{
    CoordType d1, d2;
    d1 = d2 = 0;

    CoordType intersection;

    intersection = calcIntersection(corners[0][0], corners[1][0], distance[0], distance[1]);
    positions[0] = Vertex<CoordType>(intersection, corners[0][1], corners[0][2]);

    intersection = calcIntersection(corners[1][1], corners[2][1], distance[1], distance[2]);
    positions[1] = Vertex<CoordType>(corners[1][0], intersection, corners[1][2]);

    intersection = calcIntersection(corners[3][0], corners[2][0], distance[3], distance[2]);
    positions[2] = Vertex<CoordType>(intersection, corners[2][1], corners[2][2]);

    intersection = calcIntersection(corners[0][1], corners[3][1], distance[0], distance[3]);
    positions[3] = Vertex<CoordType>(corners[3][0], intersection, corners[3][2]);

    //Back Quad
    intersection = calcIntersection(corners[4][0], corners[5][0], distance[4], distance[5]);
    positions[4] = Vertex<CoordType>(intersection, corners[4][1], corners[4][2]);

    intersection = calcIntersection(corners[5][1], corners[6][1], distance[5], distance[6]);
    positions[5] = Vertex<CoordType>(corners[5][0], intersection, corners[5][2]);


    intersection = calcIntersection(corners[7][0], corners[6][0], distance[7], distance[6]);
    positions[6] = Vertex<CoordType>(intersection, corners[6][1], corners[6][2]);

    intersection = calcIntersection(corners[4][1], corners[7][1], distance[4], distance[7]);
    positions[7] = Vertex<CoordType>(corners[7][0], intersection, corners[7][2]);

    //Sides
    intersection = calcIntersection(corners[0][2], corners[4][2], distance[0], distance[4]);
    positions[8] = Vertex<CoordType>(corners[0][0], corners[0][1], intersection);

    intersection = calcIntersection(corners[1][2], corners[5][2], distance[1], distance[5]);
    positions[9] = Vertex<CoordType>(corners[1][0], corners[1][1], intersection);

    intersection = calcIntersection(corners[3][2], corners[7][2], distance[3], distance[7]);
    positions[10] = Vertex<CoordType>(corners[3][0], corners[3][1], intersection);

    intersection = calcIntersection(corners[2][2], corners[6][2], distance[2], distance[6]);
    positions[11] = Vertex<CoordType>(corners[2][0], corners[2][1], intersection);

}


template<typename CoordType, typename IndexType>
void FastBox<CoordType, IndexType>::getSurface(BaseMesh<Vertex<CoordType>, IndexType> &mesh,
                                               vector<QueryPoint<CoordType> > &qp,
                                               IndexType &globalIndex)
{
    Vertex<CoordType> corners[8];
    Vertex<CoordType> vertex_positions[12];
    Vertex<CoordType> tmp_vertices[12];

    CoordType distances[8];

    getCorners(corners, qp);
    getDistances(distances, qp);
    getIntersections(corners, distances, vertex_positions);

    int index = getIndex(qp);
    int edge_index = 0;
    int vertex_count = 0;
    int tmp_indices[12];

    Vertex<CoordType> diff1, diff2;
    Normal<CoordType> normal;

    int current_index = 0;
    int triangle_indices[3];

    for(int a = 0; MCTable[index][a] != -1; a+= 3){
        for(int b = 0; b < 3; b++){
            edge_index = MCTable[index][a + b];
            current_index = -1;

            //If current vertex index doesn't exist
            //look for it in the suitable neighbor boxes
            if(m_intersections[edge_index] == -1){
                for(int i = 0; i < 3; i++){
                    FastBox* current_neighbor = m_neighbors[neighbor_table[edge_index][i]];

                    //If neighbor exists search for suitable index
                    if(current_neighbor != 0){
                        if(current_neighbor->m_intersections[neighbor_vertex_table[edge_index][i]] != -1){
                            current_index = current_neighbor->m_intersections[neighbor_vertex_table[edge_index][i]];
                        }
                    }
                }
            }

            //If no index was found generate new index and vertex
            //and update all neighbor boxes
            if(current_index == -1){
                m_intersections[edge_index] = globalIndex;
                Vertex<CoordType> v = vertex_positions[edge_index];
                mesh.addVertex(v);
                //mesh.addNormal(Normal<CoordType>());
                for(int i = 0; i < 3; i++){
                    FastBox* current_neighbor = m_neighbors[neighbor_table[edge_index][i]];
                    if(current_neighbor != 0){
                        current_neighbor->m_intersections[neighbor_vertex_table[edge_index][i]] = globalIndex;
                    }
                }
                globalIndex++;
            } else {
                m_intersections[edge_index] = current_index;
            }

            //Save vertices and indices for normal calculation
            tmp_vertices[vertex_count] = vertex_positions[edge_index];
            tmp_indices[vertex_count]  = m_intersections[edge_index];

            //Save vertex index in mesh
            //mesh.addIndex(intersections[edge_index]);
            triangle_indices[b] = m_intersections[edge_index];
            //Count generated vertices
            vertex_count++;
        }
        mesh.addTriangle(triangle_indices[0], triangle_indices[1], triangle_indices[2]);
    }

    //Calculate normals
    for(int i = 0; i < vertex_count - 2; i+= 3){
        diff1 = tmp_vertices[i] - tmp_vertices[i+1];
        diff2 = tmp_vertices[i+1] - tmp_vertices[i+2];
        normal = diff1.cross(diff2);

        //Interpolate with normals in mesh
        for(int j = 0; j < 3; j++){
            //mesh.interpolateNormal( normal, tmp_indices[i+j]);
        }
    }

    //return globalIndex;
}

} // namespace lssr
