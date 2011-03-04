/*
 * FastReconstruction.cpp
 *
 *  Created on: 16.02.2011
 *      Author: Thomas Wiemann
 */
#include "BaseMesh.hpp"

#include <ext/hash_map>
using namespace __gnu_cxx;

namespace lssr
{

const static int shared_vertex_table[8][28] = {
    {-1, 0, 0, 1, -1, -1, 0, 2,  0, -1, 0, 3, -1,  0, -1, 5, -1, -1, -1, 6,  0, -1, -1, 7,  0,  0, -1, 4},
    { 1, 0, 0, 0,  1, -1, 0, 3,  0, -1, 0, 2,  0,  0, -1, 5,  1,  0, -1, 4,  1, -1, -1, 7,  0, -1, -1, 6},
    { 1, 1, 0, 0,  0,  1, 0, 1,  1,  0, 0, 3,  1,  1, -1, 4,  0,  1, -1, 5,  0,  0, -1, 6,  1,  0, -1, 7},
    { 0, 1, 0, 0, -1,  1, 0, 1, -1,  0, 0, 2,  0,  1, -1, 4, -1,  1, -1, 5, -1,  0, -1, 6,  0,  0, -1, 7},
    { 0, 0, 1, 0, -1,  0, 1, 1, -1, -1, 1, 2,  0, -1,  1, 3, -1,  0,  0, 5, -1, -1,  0, 6,  0, -1,  0, 7},
    { 1, 0, 1, 0,  0,  0, 1, 1,  0, -1, 1, 2,  1, -1,  1, 3,  1,  0,  0, 4,  0, -1,  0, 6,  1, -1,  0, 7},
    { 1, 1, 1, 0,  0,  1, 1, 1,  0,  0, 1, 2,  1,  0,  1, 3,  1,  1,  0, 4,  0,  1,  0, 5,  1,  0,  0, 7},
    { 0, 1, 1, 0, -1,  1, 1, 1, -1,  0, 1, 2,  0,  0,  1, 3,  0,  1,  0, 4, -1,  1,  0, 5, -1,  0,  0, 6}
};


//This table states where each coordinate of a box vertex is relatively
//to the box center
const static int box_creation_table[8][3] = {
    {-1, -1, -1},
    { 1, -1, -1},
    { 1,  1, -1},
    {-1,  1, -1},
    {-1, -1,  1},
    { 1, -1,  1},
    { 1,  1,  1},
    {-1,  1,  1}
};

template<typename CoordType, typename IndexType>
FastReconstruction<CoordType, IndexType>::FastReconstruction(PointCloudManager<CoordType> &manager,  int resolution)
    : Reconstructor<CoordType, IndexType>(manager)
{
    // Determine m_voxelsize
    assert(resolution > 0);
    BoundingBox<CoordType> bb = this->m_manager.getBoundingBox();
    assert(bb.isValid());
    m_voxelsize = (CoordType) bb.getLongestSide() / resolution;

    // Calculate max grid indices
    calcIndices();


}


template<typename CoordType, typename IndexType>
void FastReconstruction<CoordType, IndexType>::calcIndices()
{
    BoundingBox<CoordType> bb = this->m_manager.getBoundingBox();

    CoordType max_size = bb.getLongestSide();

    //Save needed grid parameters
    m_maxIndex = (int)ceil( (max_size + 5 * m_voxelsize) / m_voxelsize);
    m_maxIndexSquare = m_maxIndex * m_maxIndex;

    m_maxIndexX = (int)ceil(bb.getXSize() / m_voxelsize) + 1;
    m_maxIndexY = (int)ceil(bb.getYSize() / m_voxelsize) + 2;
    m_maxIndexZ = (int)ceil(bb.getZSize() / m_voxelsize) + 3;
}

template<typename CoordType, typename IndexType>
void FastReconstruction<CoordType, IndexType>::getMesh(BaseMesh<Vertex<CoordType>, IndexType> &mesh)
{
    cout << timestamp << "Creating Grid..." << endl;

    //Current indices
    int index_x, index_y, index_z;
    int hash_value;

    float vsh = 0.5 * m_voxelsize;

    //Iterators
    typename hash_map<int, FastBox<CoordType, IndexType>* >::iterator it;
    typename hash_map<int, FastBox<CoordType, IndexType>* >::iterator neighbor_it;

    int global_index = 0;
    int current_index = 0;

    int dx, dy, dz;

    for(size_t i = 0; i < ; i++){
        index_x = calcIndex((points[i][0] - bounding_box.v_min.x) / voxelsize);
        index_y = calcIndex((points[i][1] - bounding_box.v_min.y) / voxelsize);
        index_z = calcIndex((points[i][2] - bounding_box.v_min.z) / voxelsize);


        for(int j = 0; j < 8; j++){

            dx = HGCreateTable[j][0];
            dy = HGCreateTable[j][1];
            dz = HGCreateTable[j][2];

            hash_value = hashValue(index_x + dx, index_y + dy, index_z +dz);
            it = cells.find(hash_value);
            if(it == cells.end()){
                //Calculate box center
                Vertex box_center = Vertex((index_x + dx) * voxelsize + bounding_box.v_min.x,
                                           (index_y + dy) * voxelsize + bounding_box.v_min.y,
                                           (index_z + dz) * voxelsize + bounding_box.v_min.z);

                //Create new box
                FastBox* box = new FastBox;

                //Setup the box itself
                for(int k = 0; k < 8; k++){

                    //Find point in Grid
                    current_index = findQueryPoint(k, index_x + dx, index_y + dy, index_z + dz);

                    //If point exist, save index in box
                    if(current_index != -1) box->vertices[k] = current_index;
                    //Otherwise create new grid point and associate it with the current box
                    else{
                        Vertex position(box_center.x + box_creation_table[k][0] * vsh,
                                box_center.y + box_creation_table[k][1] * vsh,
                                box_center.z + box_creation_table[k][2] * vsh);

                        query_points.push_back(QueryPoint(position));

                        box->vertices[k] = global_index;
                        global_index++;

                    }
                }

                //Set pointers to the neighbors of the current box
                int neighbor_index = 0;
                int neighbor_hash = 0;

                for(int a = -1; a < 2; a++){
                    for(int b = -1; b < 2; b++){
                        for(int c = -1; c < 2; c++){

                            //Calculate hash value for current neighbor cell
                            neighbor_hash = hashValue(index_x + dx + a,
                                                      index_y + dy + b,
                                                      index_z + dz + c);

                            //Try to find this cell in the grid
                            neighbor_it = cells.find(neighbor_hash);

                            //If it exists, save pointer in box
                            if(neighbor_it != cells.end()){
                                box->neighbors[neighbor_index] = (*neighbor_it).second;
                            }

                            neighbor_index++;
                        }
                    }
                }

                cells[hash_value] = box;
            }
        }
    }
    cout << timestamp << "Finished Grid Creation. Number of generated cells:        " << cells.size() << endl;
    cout << timestamp << "Finished Grid Creation. Number of generated query points: " << query_points.size() << endl;

}

template<typename CoordType, typename IndexType>
void FastReconstruction<CoordType, IndexType>::createGrid()
{

}

} //namespace lssr
