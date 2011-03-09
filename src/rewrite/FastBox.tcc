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
                                               vector<QueryPoint<CoordType> > &query_points)
{

}

template<typename CoordType, typename IndexType>
void FastBox<CoordType, IndexType>::getDistances(CoordType distances[],
                                                 vector<QueryPoint<CoordType> > &query_points)
{

}


template<typename CoordType, typename IndexType>
void FastBox<CoordType, IndexType>::getSurface(BaseMesh<Vertex<CoordType>, IndexType> &mesh,
                                               vector<QueryPoint<CoordType> > &qp,
                                               IndexType &globalIndex)
{
    Vertex<CoordType> corners[8];
    Vertex<CoordType> vertex_positions[12];
    Vertex<CoordType> tmp_vertices[12];

    IndexType distances[8];

    getCorners(corners, qp);
    getDistances(distances, qp);
    getIntersections(corners, distances, vertex_positions);

    int index = getIndex();
    int edge_index = 0;
    int vertex_count = 0;
    int tmp_indices[12];

    BaseVertex diff1, diff2;
    Normal normal;

    int current_index = 0;
    int triangle_indices[3];

    for(int a = 0; MCTable[index][a] != -1; a+= 3){
        for(int b = 0; b < 3; b++){
            edge_index = MCTable[index][a + b];
            current_index = -1;

            //If current vertex index doesn't exist
            //look for it in the suitable neighbor boxes
            if(intersections[edge_index] == -1){
                for(int i = 0; i < 3; i++){
                    FastBox* current_neighbor = neighbors[neighbor_table[edge_index][i]];

                    //If neighbor exists search for suitable index
                    if(current_neighbor != 0){
                        if(current_neighbor->intersections[neighbor_vertex_table[edge_index][i]] != -1){
                            current_index = current_neighbor->intersections[neighbor_vertex_table[edge_index][i]];
                        }
                    }
                }
            }

            //If no index was found generate new index and vertex
            //and update all neighbor boxes
            if(current_index == -1){
                intersections[edge_index] = global_index;
                ColorVertex v = vertex_positions[edge_index];
                mesh.addVertex(v);
                mesh.addNormal(Normal());
                for(int i = 0; i < 3; i++){
                    FastBox* current_neighbor = neighbors[neighbor_table[edge_index][i]];
                    if(current_neighbor != 0){
                        current_neighbor->intersections[neighbor_vertex_table[edge_index][i]] = global_index;
                    }
                }
                global_index++;
            } else {
                intersections[edge_index] = current_index;
            }

            //Save vertices and indices for normal calculation
            tmp_vertices[vertex_count] = vertex_positions[edge_index];
            tmp_indices[vertex_count]  = intersections[edge_index];

            //Save vertex index in mesh
            //mesh.addIndex(intersections[edge_index]);
            triangle_indices[b] = intersections[edge_index];
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
            mesh.interpolateNormal( normal, tmp_indices[i+j]);
        }
    }

    return global_index;
}

} // namespace lssr
