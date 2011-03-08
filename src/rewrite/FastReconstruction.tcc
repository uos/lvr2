/*
 * FastReconstruction.cpp
 *
 *  Created on: 16.02.2011
 *      Author: Thomas Wiemann
 */
#include "BaseMesh.hpp"
#include "FastReconstructionTables.hpp"

namespace lssr
{

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
IndexType FastReconstruction<CoordType, IndexType>::findQueryPoint(
        const int &position, const int &x, const int &y, const int &z)
{
    int n_x, n_y, n_z, q_v, offset;
    typename hash_map<size_t, FastBox<CoordType, IndexType>* >::iterator it;

    for(int i = 0; i < 7; i++){
        offset = i * 4;
        n_x = x + shared_vertex_table[position][offset];
        n_y = y + shared_vertex_table[position][offset + 1];
        n_z = z + shared_vertex_table[position][offset + 2];
        q_v = shared_vertex_table[position][offset + 3];

   //     size_t hash = hashValue(n_x, n_y, n_z);

        it = m_cells.find(0);
        if(it != m_cells.end())
        {
            FastBox<CoordType, IndexType>* b = it->second;
            if(b->getVertex(q_v) != -1) return b->getVertex(q_v);
        }
    }

    return -1;


}

template<typename CoordType, typename IndexType>
void FastReconstruction<CoordType, IndexType>::getMesh(BaseMesh<Vertex<CoordType>, IndexType> &mesh)
{
}

template<typename CoordType, typename IndexType>
void FastReconstruction<CoordType, IndexType>::createGrid()
{
    cout << timestamp << "Creating Grid..." << endl;

       //  Needed local variables
       int index_x, index_y, index_z;
       size_t hash_value;

       float vsh = 0.5 * m_voxelsize;

       // Some iterators for hash map accesses
       typename hash_map<size_t, FastBox<CoordType, IndexType>* >::iterator it;
       typename hash_map<size_t, FastBox<CoordType, IndexType>* >::iterator neighbor_it;

       // Values for current and global indices. Current refers to a
       // already present query point, global index is id that the next
       // created query point will get
       int global_index = 0;
       int current_index = 0;

       int dx, dy, dz;

       // Get min and max vertex of the point clouds bounding box
       BoundingBox<CoordType> bounding_box = this->m_manager.getBoundingBox();
       Vertex<CoordType> v_min = bounding_box.getMin();
       Vertex<CoordType> v_max = bounding_box.getMax();

       for(size_t i = 0; i < this->m_manager.getNumPoints(); i++)
       {
           /// TODO: Replace with Vertex<> ???
           index_x = calcIndex((this->m_manager[i][0] - v_min[0]) / m_voxelsize);
           index_y = calcIndex((this->m_manager[i][1] - v_min[1]) / m_voxelsize);
           index_z = calcIndex((this->m_manager[i][2] - v_min[2]) / m_voxelsize);


           for(int j = 0; j < 8; j++){

               // Get the grid offsets for the neighboring grid position
               // for the given box corner
               dx = HGCreateTable[j][0];
               dy = HGCreateTable[j][1];
               dz = HGCreateTable[j][2];

               hash_value = hashValue(index_x + dx, index_y + dy, index_z +dz);


               it = m_cells.find(hash_value);
               if(it == m_cells.end()){
                   //Calculate box center
                   Vertex<CoordType> box_center((index_x + dx) * m_voxelsize + v_min[0],
                                                (index_y + dy) * m_voxelsize + v_min[1],
                                                (index_z + dz) * m_voxelsize + v_min[2]);

                   //Create new box
                   FastBox<CoordType, IndexType>* box = new FastBox<CoordType, IndexType>(box_center);

                   //Setup the box itself
                   for(int k = 0; k < 8; k++){

                       //Find point in Grid
                       current_index = findQueryPoint(k, index_x + dx, index_y + dy, index_z + dz);

                       //If point exist, save index in box
                       if(current_index != -1) box->setVertex(k, current_index);

                       //Otherwise create new grid point and associate it with the current box
                       else{
                           Vertex<CoordType> position(box_center[0] + box_creation_table[k][0] * vsh,
                                                      box_center[1] + box_creation_table[k][1] * vsh,
                                                      box_center[2] + box_creation_table[k][2] * vsh);

                           m_queryPoints.push_back(QueryPoint<CoordType>(position));

                           box->setVertex(k, global_index);
                           global_index++;

                       }
                   }

                   //Set pointers to the neighbors of the current box
                   int neighbor_index = 0;
                   int neighbor_hash = 0;

                   for(int a = -1; a < 2; a++)
                   {
                       for(int b = -1; b < 2; b++)
                       {
                           for(int c = -1; c < 2; c++)
                           {

                               //Calculate hash value for current neighbor cell
                               neighbor_hash = hashValue(index_x + dx + a,
                                                         index_y + dy + b,
                                                         index_z + dz + c);

                               //Try to find this cell in the grid
                               neighbor_it = m_cells.find(neighbor_hash);

                               //If it exists, save pointer in box
                               if(neighbor_it != m_cells.end())
                               {
                                   box->setNeighbor(neighbor_index, (*neighbor_it).second);
                               }

                               neighbor_index++;
                           }
                       }
                   }

                   m_cells[hash_value] = box;
               }
           }
       }
       cout << timestamp << "Finished Grid Creation. Number of generated cells:        " << m_cells.size() << endl;
       cout << timestamp << "Finished Grid Creation. Number of generated query points: " << m_queryPoints.size() << endl;

}

} //namespace lssr
