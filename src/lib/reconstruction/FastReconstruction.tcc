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
 * FastReconstruction.cpp
 *
 *  Created on: 16.02.2011
 *      Author: Thomas Wiemann
 */
#include "../geometry/BaseMesh.hpp"
#include "../reconstruction/FastReconstructionTables.hpp"

namespace lssr
{

template<typename VertexT, typename NormalT>
FastReconstruction<VertexT, NormalT>::FastReconstruction(
        typename PointCloudManager<VertexT, NormalT>::Ptr manager,
        float resolution,
        bool isVoxelsize,
        bool useTetraeder)
    : Reconstructor<VertexT, NormalT>(manager), m_useTetraeder(useTetraeder)
{
    // Determine m_voxelsize
    assert(resolution > 0);
    BoundingBox<VertexT> bb = this->m_manager->getBoundingBox();

    assert(bb.isValid());
    if(!isVoxelsize)
    {
        m_voxelsize = (float) bb.getLongestSide() / resolution;
    }
    else
    {
        m_voxelsize = resolution;
    }

    FastBox<VertexT, NormalT>::m_voxelsize = m_voxelsize;

    // Calculate max grid indices
    calcIndices();
    createGrid();
    calcQueryPointValues();


}

template<typename VertexT, typename NormalT>
void FastReconstruction<VertexT, NormalT>::calcIndices()
{
    BoundingBox<VertexT> bb = this->m_manager->getBoundingBox();

    float max_size = bb.getLongestSide();

    //Save needed grid parameters
    m_maxIndex = (int)ceil( (max_size + 5 * m_voxelsize) / m_voxelsize);
    m_maxIndexSquare = m_maxIndex * m_maxIndex;

    m_maxIndexX = (int)ceil(bb.getXSize() / m_voxelsize) + 1;
    m_maxIndexY = (int)ceil(bb.getYSize() / m_voxelsize) + 2;
    m_maxIndexZ = (int)ceil(bb.getZSize() / m_voxelsize) + 3;
}

template<typename VertexT, typename NormalT>
uint FastReconstruction<VertexT, NormalT>::findQueryPoint(
        const int &position, const int &x, const int &y, const int &z)
{
    int n_x, n_y, n_z, q_v, offset;
    typename hash_map<size_t, FastBox<VertexT, NormalT>* >::iterator it;

    for(int i = 0; i < 7; i++){
        offset = i * 4;
        n_x = x + shared_vertex_table[position][offset];
        n_y = y + shared_vertex_table[position][offset + 1];
        n_z = z + shared_vertex_table[position][offset + 2];
        q_v = shared_vertex_table[position][offset + 3];

        size_t hash = hashValue(n_x, n_y, n_z);

        it = m_cells.find(hash);
        if(it != m_cells.end())
        {
            FastBox<VertexT, NormalT>* b = it->second;
            if(b->getVertex(q_v) != FastBox<VertexT, NormalT>::INVALID_INDEX) return b->getVertex(q_v);
        }
    }

    return FastBox<float, uint>::INVALID_INDEX;


}

template<typename VertexT, typename NormalT>
void FastReconstruction<VertexT, NormalT>::createGrid()
{
	cout << timestamp << "Creating Grid..." << endl;

	//  Needed local variables
	int index_x, index_y, index_z;
	size_t hash_value;

	uint INVALID = FastBox<float, uint>::INVALID_INDEX;

	float vsh = 0.5 * m_voxelsize;

	// Some iterators for hash map accesses
	typename hash_map<size_t, FastBox<VertexT, NormalT>* >::iterator it;
	typename hash_map<size_t, FastBox<VertexT, NormalT>* >::iterator neighbor_it;

	// Values for current and global indices. Current refers to a
	// already present query point, global index is id that the next
	// created query point will get
	int global_index = 0;
	uint current_index = 0;

	int dx, dy, dz;

	// Get min and max vertex of the point clouds bounding box
	BoundingBox<VertexT> bounding_box = this->m_manager->getBoundingBox();
	VertexT v_min = bounding_box.getMin();
	VertexT v_max = bounding_box.getMax();

	for(size_t i = 0; i < this->m_manager->getNumPoints(); i++)
	{
        VertexT currentVertex = this->m_manager->getPoint(i);
		/// TODO: Replace with Vertex<> ???
		index_x = calcIndex((currentVertex[0] - v_min[0]) / m_voxelsize);
		index_y = calcIndex((currentVertex[1] - v_min[1]) / m_voxelsize);
		index_z = calcIndex((currentVertex[2] - v_min[2]) / m_voxelsize);
		//index_x = calcIndex((this->m_manager[i][0] - v_min[0]) / m_voxelsize);
		//index_y = calcIndex((this->m_manager[i][1] - v_min[1]) / m_voxelsize);
		//index_z = calcIndex((this->m_manager[i][2] - v_min[2]) / m_voxelsize);


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
				VertexT box_center((index_x + dx) * m_voxelsize + v_min[0],
						(index_y + dy) * m_voxelsize + v_min[1],
						(index_z + dz) * m_voxelsize + v_min[2]);

				//Create new box
				FastBox<VertexT, NormalT>* box = 0;
				if(!m_useTetraeder)
				{
				    box = new FastBox<VertexT, NormalT>(box_center);
				}
				else
				{
				    box = new TetraederBox<VertexT, NormalT>(box_center);
				}

				//Setup the box itself
				for(int k = 0; k < 8; k++){

					//Find point in Grid
					current_index = findQueryPoint(k, index_x + dx, index_y + dy, index_z + dz);

					//If point exist, save index in box
					if(current_index != INVALID) box->setVertex(k, current_index);

					//Otherwise create new grid point and associate it with the current box
					else{
						VertexT position(box_center[0] + box_creation_table[k][0] * vsh,
								box_center[1] + box_creation_table[k][1] * vsh,
								box_center[2] + box_creation_table[k][2] * vsh);

						m_queryPoints.push_back(QueryPoint<VertexT>(position));

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
								(*neighbor_it).second->setNeighbor(26 - neighbor_index, box);
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


template<typename VertexT, typename NormalT>
void FastReconstruction<VertexT, NormalT>::getMesh(BaseMesh<VertexT, NormalT> &mesh)
{
	// Status message for mesh generation
	string comment = timestamp.getElapsedTime() + "Creating Mesh ";
	ProgressBar progress(m_cells.size(), comment);

	// Some pointers
	FastBox<VertexT, NormalT>* b;
	uint global_index = 0;

	// Iterate through cells and calculate local approximations
	typename hash_map<size_t, FastBox<VertexT, NormalT>* >::iterator it;
	for(it = m_cells.begin(); it != m_cells.end(); it++){
		b = it->second;
		b->getSurface(mesh, m_queryPoints, global_index);
		++progress;
	}
	cout << endl;
}

template<typename VertexT, typename NormalT>
void FastReconstruction<VertexT, NormalT>::calcQueryPointValues(){

    // Status message output
    string comment = timestamp.getElapsedTime() + "Calculating distance values ";
    ProgressBar progress((int)m_queryPoints.size(), comment);

    Timestamp ts;

    // Calculate a distance value for each query point
    #pragma omp parallel for
    for(size_t i = 0; i < m_queryPoints.size(); i++){
        float projectedDistance;
        float euklideanDistance;

        //cout << euklideanDistance << " " << projectedDistance << endl;

        this->m_manager->distance(m_queryPoints[i].m_position, projectedDistance, euklideanDistance);
        if (euklideanDistance > 1.7320 * m_voxelsize)
        {
        	m_queryPoints[i].m_invalid = true;
        }
        m_queryPoints[i].m_distance = projectedDistance;
        ++progress;
    }

    cout << endl;
    cout << timestamp << "Elapsed time: " << ts << endl;
}


} //namespace lssr
