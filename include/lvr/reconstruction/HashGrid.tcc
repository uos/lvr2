/*
 * HashGrid.cpp
 *
 *  Created on: Nov 27, 2014
 *      Author: twiemann
 */



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
 * HashGrid.cpp
 *
 *  Created on: 16.02.2011
 *      Author: Thomas Wiemann
 */
#include "HashGrid.hpp"
#include <lvr/geometry/BaseMesh.hpp>
#include "FastReconstructionTables.hpp"
#include "SharpBox.hpp"
#include <lvr/io/Progress.hpp>

namespace lvr
{

template<typename VertexT, typename BoxT>
HashGrid<VertexT, BoxT>::HashGrid(float cellSize, BoundingBox<VertexT> boundingBox, bool isVoxelsize, bool extrude) :
	GridBase(extrude),
	m_boundingBox(boundingBox),
	m_globalIndex(0)
{
	m_coordinateScales[0] = 1.0;
	m_coordinateScales[1] = 1.0;
	m_coordinateScales[2] = 1.0;

	if(!m_boundingBox.isValid())
	{
		cout << timestamp << "Warning: Malformed BoundingBox." << endl;
	}

	if(!isVoxelsize)
	{
		m_voxelsize = (float) m_boundingBox.getLongestSide() / cellSize;
	}
	else
	{
		m_voxelsize = cellSize;
	}

	cout << timestamp << "Used voxelsize is " << m_voxelsize << endl;

	if(!m_extrude)
	{
		cout << timestamp << "Grid is not extruded." << endl;
	}


	BoxT::m_voxelsize = m_voxelsize;
	calcIndices();
}

template<typename VertexT, typename BoxT>
HashGrid<VertexT, BoxT>::HashGrid(string file)
{



	ifstream ifs(file.c_str());
	float minx, miny, minz, maxx, maxy, maxz, vsize;
	size_t qsize, csize;
	ifs >> minx >> miny >> minz >> maxx >> maxy >> maxz >> qsize >> vsize >> csize;

	m_boundingBox = BoundingBox<VertexT>(minx, miny, minz, maxx, maxy, maxz);
	m_globalIndex = 0;
	m_extrude=false; // TODO: ADD TO SERIALIZATION
	m_coordinateScales[0] = 1.0;
	m_coordinateScales[1] = 1.0;
	m_coordinateScales[2] = 1.0;
	m_voxelsize = vsize;
	BoxT::m_voxelsize = m_voxelsize;
	calcIndices();


	float  pdist;
	VertexT v;
	//cout << timestamp << "Creating Grid..." << endl;

	// Iterator over all points, calc lattice indices and add lattice points to the grid
	for(size_t i = 0; i < qsize; i++)
	{

		ifs >> v[0] >> v[1] >> v[2] >> pdist;

		QueryPoint<VertexT> qp(v, pdist);
		m_queryPoints.push_back(qp);

	}
	//cout << timestamp << "read qpoints.. csize: " << csize << endl;
	size_t h;
	unsigned int cell[8];
	VertexT cell_center;
	for(size_t k = 0 ; k< csize ; k++)
	{
		//cout << "i: " << k << endl;
		ifs >> h >> cell[0] >> cell[1] >> cell[2] >> cell[3] >> cell[4] >> cell[5] >> cell[6] >> cell[7]
				 >> cell_center[0] >> cell_center[1] >> cell_center[2] ;
		//cout <<  h << " " << cell[0] << " " << cell[1] << " " << cell[2] << " " << cell[3] << " " << cell[4] << " " << cell[5] << " " << cell[6] << " " << cell[7]
		//		 << " " << cell_center[0] << " " << cell_center[1] << " " << cell_center[2] << endl;
		BoxT* box = new BoxT(cell_center);
		for(int j=0 ; j<8 ; j++)
		{
			box->setVertex(j,  cell[j]);
		}

		m_cells[h] = box;
	}
	cout << timestamp << "Reading cells.." << endl;
	typename HashGrid<VertexT, BoxT>::box_map_it it;
	typename HashGrid<VertexT, BoxT>::box_map_it neighbor_it;

	cout << "c size: " << m_cells.size() << endl;
	for( it = m_cells.begin() ; it != m_cells.end() ; it++)
	{
		//cout << "asdfsdfgsdfgdsfgdfg" << endl;
		BoxT* currentBox = it->second;
		int neighbor_index = 0;
		size_t neighbor_hash = 0;

		for(int a = -1; a < 2; a++)
		{
			for(int b = -1; b < 2; b++)
			{
				for(int c = -1; c < 2; c++)
				{


					//Calculate hash value for current neighbor cell
					neighbor_hash = this->hashValue(it->first + a,
													it->first + b,
													it->first + c);
					//cout << "n hash: " << neighbor_hash  << endl;
					//cout << " id: " << neighbor_index << endl;

					//Try to find this cell in the grid
					neighbor_it = this->m_cells.find(neighbor_hash);

					//If it exists, save pointer in box
					if(neighbor_it != this->m_cells.end())
					{
						currentBox->setNeighbor(neighbor_index, (*neighbor_it).second);
						(*neighbor_it).second->setNeighbor(26 - neighbor_index, currentBox);
					}

					neighbor_index++;
				}
			}
		}
	}
	cout << "Finished reading grid" << endl;


}

template<typename VertexT, typename BoxT>
bool HashGrid<VertexT, BoxT>::interpolateEdge(HashGrid<VertexT, BoxT>& rhs)
{
    //Calc intersection Bounding box
    Vertexf ibb_min, ibb_max;

    Vertexf addV_lhs(m_voxelsize, m_voxelsize,m_voxelsize);
    Vertexf lhs_bb_min = this->m_boundingBox.getMin();
    Vertexf lhs_bb_max = this->m_boundingBox.getMax();
    Vertexf rhs_bb_min = rhs.m_boundingBox.getMin();
    Vertexf rhs_bb_max = rhs.m_boundingBox.getMax();
    lhs_bb_min-=addV_lhs;
    lhs_bb_max+=addV_lhs;
    rhs_bb_min-=addV_lhs;
    rhs_bb_max+=addV_lhs;
    if(m_extrude)
    {
        lhs_bb_min-=addV_lhs;
        lhs_bb_max+=addV_lhs;
        rhs_bb_min-=addV_lhs;
        rhs_bb_max+=addV_lhs;
    }

    ibb_min.x = std::max(lhs_bb_min.x, rhs_bb_min.x);
    ibb_min.y = std::max(lhs_bb_min.y, rhs_bb_min.y);
    ibb_min.z = std::max(lhs_bb_min.z, rhs_bb_min.z);
    ibb_max.x = std::min(lhs_bb_max.x, rhs_bb_max.x);
    ibb_max.y = std::min(lhs_bb_max.y, rhs_bb_max.y);
    ibb_max.z = std::min(lhs_bb_max.z, rhs_bb_max.z);

    int maxX = calcIndex((ibb_max.x - ibb_min.x)/m_voxelsize);
    int maxY = calcIndex((ibb_max.y - ibb_min.y)/m_voxelsize);
    int maxZ = calcIndex((ibb_max.z - ibb_min.z)/m_voxelsize);
//    cout << "bb-this: " << lhs_bb_min << endl <<lhs_bb_max << endl << "bb-rhs: " << rhs_bb_min << endl << rhs_bb_max << endl ;
//    cout << "ibb:" << ibb_min << endl << ibb_max << endl;
//    cout << "max ibb indices: " << maxX << "|" << maxY << "|" << maxZ << endl;
    if(m_extrude)
    {
        maxX++;
        maxY++;
        maxZ++;
    }
    Vertexf middleAdd(m_voxelsize/2, m_voxelsize/2, m_voxelsize/2);
    for(int i = 0 ; i < maxX ; i++)
    {
        for(int j = 0 ; j < maxY; j++)
        {
            for(int k = 0 ; k < maxZ ; k++)
            {
                Vertexf checkPoint = ibb_min;
                checkPoint.x += (i * m_voxelsize + (m_voxelsize/4)) ;
                checkPoint.y += (j * m_voxelsize)+ (m_voxelsize/4) ;
                checkPoint.z += (k * m_voxelsize) + (m_voxelsize/4);

                Vertex<int> lhs_coord = this->convertGlobalToGrid(checkPoint);
                Vertex<int> rhs_coord = rhs.convertGlobalToGrid(checkPoint);
//                cout << "global: " << endl << checkPoint << endl;
//                cout << "coords: " << endl << lhs_coord << endl << rhs_coord << endl;
                box_map_it lhs_it = m_cells.find(this->hashValue(lhs_coord[0], lhs_coord[1], lhs_coord[2]));
                if(lhs_it == m_cells.end()) continue;

                box_map_it rhs_it = rhs.m_cells.find(rhs.hashValue(rhs_coord[0], rhs_coord[1], rhs_coord[2]));
                if(rhs_it == rhs.m_cells.end()) continue;

                BoxT* lhs_box = lhs_it->second;
                BoxT* rhs_box = rhs_it->second;


                /*    if(lhs_box->getCenter() != rhs_box->getCenter()) continue;*/
                if(        !((fabs(lhs_box->getCenter().x - rhs_box->getCenter().x) <= (m_voxelsize/4)) &&
                        (fabs(lhs_box->getCenter().y - rhs_box->getCenter().y) <= (m_voxelsize/4)) &&
                        (fabs(lhs_box->getCenter().z - rhs_box->getCenter().z) <= (m_voxelsize/4)))) continue;
                //float dist[8];
//                if(lhs_box->m_extruded || rhs_box->m_extruded) continue;
//                cout << "box centers: " << endl << lhs_box->getCenter() << endl << rhs_box->getCenter() << endl;
//                rhs_box->m_overlappBox = true;
//                lhs_box->m_overlappBox = true;
                bool all_checked = true;
                float lhs_original_dist[8];
                float rhs_original_dist[8];
                for(int m = 0 ; m<8;m++)
                {
                    lhs_original_dist[m] = m_queryPoints[lhs_box->getVertex(m)].m_position.x;
                    rhs_original_dist[m] = rhs.m_queryPoints[rhs_box->getVertex(m)].m_position.x;
                }

                for(int l = 0 ; l<8; l++)
                {
//                    cout << "interpolating : " << m_queryPoints[lhs_box->getVertex(l)].m_position << " and " <<  rhs.m_queryPoints[rhs_box->getVertex(l)].m_position << endl;
//                    cout << "distances : " << m_queryPoints[lhs_box->getVertex(l)].m_distance << " and " <<  rhs.m_queryPoints[rhs_box->getVertex(l)].m_distance << endl;
//                    m_queryPoints[lhs_box->getVertex(l)].m_distance = (m_queryPoints[lhs_box->getVertex(l)].m_distance + rhs.m_queryPoints[rhs_box->getVertex(l)].m_distance) / 2;
//                    rhs.m_queryPoints[rhs_box->getVertex(l)].m_distance = m_queryPoints[lhs_box->getVertex(l)].m_distance;
//                    rhs.m_queryPoints[rhs_box->getVertex(l)].m_distance = m_queryPoints[lhs_box->getVertex(l)].m_distance;
//                    if(!((rhs.m_queryPoints[rhs_box->getVertex(l)].m_position.x - m_queryPoints[lhs_box->getVertex(l)].m_position.x <= (m_voxelsize/10)) &&
//                                (rhs.m_queryPoints[rhs_box->getVertex(l)].m_position.y - m_queryPoints[lhs_box->getVertex(l)].m_position.y <= (m_voxelsize/10)) &&
//                                (rhs.m_queryPoints[rhs_box->getVertex(l)].m_position.z - m_queryPoints[lhs_box->getVertex(l)].m_position.z <= (m_voxelsize/10))))
//                    {
//                        cout <<"nonono " << endl << rhs.m_queryPoints[rhs_box->getVertex(l)].m_position  << m_queryPoints[lhs_box->getVertex(l)].m_position << endl;
//                    }
//                    cout << "new dist: " << rhs.m_queryPoints[rhs_box->getVertex(l)].m_distance << "|" << m_queryPoints[lhs_box->getVertex(l)].m_distance << endl;
//                    if(!    ((rhs.m_queryPoints[rhs_box->getVertex(l)].m_distance < 0 && m_queryPoints[lhs_box->getVertex(l)].m_distance>0) ||
//                             (rhs.m_queryPoints[rhs_box->getVertex(l)].m_distance > 0 && m_queryPoints[lhs_box->getVertex(l)].m_distance<0) ) )
//                    {
//                        rhs.m_queryPoints[rhs_box->getVertex(l)].m_distance = (m_queryPoints[lhs_box->getVertex(l)].m_distance + rhs.m_queryPoints[rhs_box->getVertex(l)].m_distance)/2 ;
//                        m_queryPoints[lhs_box->getVertex(l)].m_distance = rhs.m_queryPoints[rhs_box->getVertex(l)].m_distance ;
//
//
//                    }
//                    else
//                    {
//                        rhs.m_queryPoints[rhs_box->getVertex(l)].m_distance = std::min(m_queryPoints[lhs_box->getVertex(l)].m_distance , rhs.m_queryPoints[rhs_box->getVertex(l)].m_distance) ;
//                        m_queryPoints[lhs_box->getVertex(l)].m_distance = rhs.m_queryPoints[rhs_box->getVertex(l)].m_distance ;
//
//                    }

                    bool found_point = false;
                    for(int m = 0 ; m<8 ; m++)
                     {
                        if((fabs(rhs.m_queryPoints[rhs_box->getVertex(l)].m_position.x - m_queryPoints[lhs_box->getVertex(m)].m_position.x) <= (m_voxelsize/10)) &&
                                (fabs(rhs.m_queryPoints[rhs_box->getVertex(l)].m_position.y - m_queryPoints[lhs_box->getVertex(m)].m_position.y) <= (m_voxelsize/10)) &&
                                (fabs(rhs.m_queryPoints[rhs_box->getVertex(l)].m_position.z - m_queryPoints[lhs_box->getVertex(m)].m_position.z) <= (m_voxelsize/10)))

                     /*                if(rhs.m_queryPoints[rhs_box->getVertex(l)].m_position == m_queryPoints[lhs_box->getVertex(m)].m_position)*/
                        {

                            if(rhs_box->m_extruded && !(lhs_box->m_extruded))
                            {
                                rhs.m_queryPoints[rhs_box->getVertex(l)].m_distance = m_queryPoints[lhs_box->getVertex(m)].m_distance;
//                                cout << "kacka0!" << endl;
                                found_point = true;
                                break;
                            }
                            else if(lhs_box->m_extruded && !(rhs_box->m_extruded))
                            {
                                m_queryPoints[lhs_box->getVertex(m)].m_distance = rhs.m_queryPoints[rhs_box->getVertex(l)].m_distance;
//                                cout << "kacka0!" << endl;
                                found_point = true;
                                break;
                            }
//                            else if(     (rhs.m_queryPoints[rhs_box->getVertex(l)].m_distance > 0 && m_queryPoints[lhs_box->getVertex(m)].m_distance < 0) ||
//                                    (rhs.m_queryPoints[rhs_box->getVertex(l)].m_distance < 0 && m_queryPoints[lhs_box->getVertex(m)].m_distance > 0))
//                            {
//
//                                rhs.m_queryPoints[rhs_box->getVertex(l)].m_distance = (m_queryPoints[lhs_box->getVertex(m)].m_distance + rhs.m_queryPoints[rhs_box->getVertex(l)].m_distance)/2 ;
//                                m_queryPoints[lhs_box->getVertex(m)].m_distance = rhs.m_queryPoints[rhs_box->getVertex(l)].m_distance ;
//                                found_point = true;
//                                break;
//                            }
                            else if(lhs_box->m_extruded && (rhs_box->m_extruded))
                            {
                                //todo: check if box was already interpolated
                                lhs_box->m_extruded = false;
                                rhs_box->m_extruded = false;
//                                cout << "kacka1!" << endl;

                                if(fabs(rhs.m_queryPoints[rhs_box->getVertex(l)].m_distance ) < fabs(m_queryPoints[lhs_box->getVertex(m)].m_distance) )
                                {
                                    m_queryPoints[lhs_box->getVertex(m)].m_distance = rhs.m_queryPoints[rhs_box->getVertex(l)].m_distance;
                                }
                                else
                                {
                                    rhs.m_queryPoints[rhs_box->getVertex(l)].m_distance = m_queryPoints[lhs_box->getVertex(m)].m_distance ;
                                }

//                                rhs.m_queryPoints[rhs_box->getVertex(l)].m_distance = (m_queryPoints[lhs_box->getVertex(m)].m_distance + rhs.m_queryPoints[rhs_box->getVertex(l)].m_distance)/2 ;
//                                m_queryPoints[lhs_box->getVertex(m)].m_distance = rhs.m_queryPoints[rhs_box->getVertex(l)].m_distance ;
                                found_point = true;
                                break;
                            }
                            else if(!(lhs_box->m_extruded) && !(rhs_box->m_extruded) )
                            {
//                                cout << "kacka2!" << endl;
                                if(fabs(rhs.m_queryPoints[rhs_box->getVertex(l)].m_distance ) < fabs(m_queryPoints[lhs_box->getVertex(m)].m_distance) )
                                {
                                    m_queryPoints[lhs_box->getVertex(m)].m_distance = rhs.m_queryPoints[rhs_box->getVertex(l)].m_distance;
                                }
                                else
                                {
                                    rhs.m_queryPoints[rhs_box->getVertex(l)].m_distance = m_queryPoints[lhs_box->getVertex(m)].m_distance ;
                                }
//                                rhs.m_queryPoints[rhs_box->getVertex(l)].m_distance = (m_queryPoints[lhs_box->getVertex(m)].m_distance + rhs.m_queryPoints[rhs_box->getVertex(l)].m_distance)/2 ;
//                                m_queryPoints[lhs_box->getVertex(m)].m_distance = rhs.m_queryPoints[rhs_box->getVertex(l)].m_distance ;
                                found_point = true;
                                break;

                            }
                            else
                            {
//                                cout << "kacka3!" << endl;
                            }
//                            else if(!    ((rhs.m_queryPoints[rhs_box->getVertex(l)].m_distance < 0 && m_queryPoints[lhs_box->getVertex(m)].m_distance>0) ||
//                                    (rhs.m_queryPoints[rhs_box->getVertex(l)].m_distance > 0 && m_queryPoints[lhs_box->getVertex(m)].m_distance<0) ) )
//                            {
//                                rhs.m_queryPoints[rhs_box->getVertex(l)].m_distance = (m_queryPoints[lhs_box->getVertex(m)].m_distance + rhs.m_queryPoints[rhs_box->getVertex(l)].m_distance)/2 ;
//                                m_queryPoints[lhs_box->getVertex(m)].m_distance = rhs.m_queryPoints[rhs_box->getVertex(l)].m_distance ;
//
//                                cout << "m/l = " << m << ", " << l << endl;
//                                cout << rhs.m_queryPoints[rhs_box->getVertex(l)].m_position << m_queryPoints[lhs_box->getVertex(m)].m_position << endl;
//                                found_point = true;
//                                break;
//                            }
//
//                            else
//                            {
//                                if(fabs(rhs.m_queryPoints[rhs_box->getVertex(l)].m_distance ) < fabs(m_queryPoints[lhs_box->getVertex(m)].m_distance) )
//                                {
//                                    m_queryPoints[lhs_box->getVertex(m)].m_distance = rhs.m_queryPoints[rhs_box->getVertex(l)].m_distance;
//                                }
//                                else
//                                {
//                                    rhs.m_queryPoints[rhs_box->getVertex(l)].m_distance = m_queryPoints[lhs_box->getVertex(m)].m_distance ;
//                                }
//                                cout << "m/l = " << m << ", " << l << endl;
//                                cout << rhs.m_queryPoints[rhs_box->getVertex(l)].m_distance << m_queryPoints[lhs_box->getVertex(m)].m_distance << endl;
////                                rhs.m_queryPoints[rhs_box->getVertex(l)].m_distance = std::min(m_queryPoints[lhs_box->getVertex(m)].m_distance , rhs.m_queryPoints[rhs_box->getVertex(l)].m_distance) ;
////                                m_queryPoints[lhs_box->getVertex(m)].m_distance = rhs.m_queryPoints[rhs_box->getVertex(l)].m_distance ;
//                                found_point = true;
//                                break;
//                            }

                        }

                    }
                    if(! found_point)
                    {
                        for(int m = 0 ; m<8;m++)
                        {
                            m_queryPoints[lhs_box->getVertex(m)].m_position.x = lhs_original_dist[m];
                            rhs.m_queryPoints[rhs_box->getVertex(l)].m_position.x = rhs_original_dist[m];
                            cout << "ohno" << endl;
                        }
                        break;
                    }


                    //if(rhs.m_queryPoints[rhs_box->getVertex(l)].m_position != m_queryPoints[lhs_box->getVertex(l)].m_position) continue;

                    /*      m_queryPoints[lhs_box->getVertex(l)].m_distance = dist[l];
      rhs.m_queryPoints[rhs_box->getVertex(l)].m_distance = dist[l];*/
                }

            }
        }
    }
}


/*
template<typename VertexT, typename BoxT>
vector<BoxT*> void HashGrid<VertexT, BoxT>::getSideCells(Vertex<int> direction)
{
	vector<BoxT*> out;
	size_t x,y,z;
	size_t  maxx, maxy, maxz;

	x = y = z = 0;
	maxx = m_maxIndexX;
	maxy = m_maxIndexY;
	maxz = m_maxIndexZ;

	if		(direction.x == 1) x = maxx = m_maxIndexX;
	else if (direction.y == 1) y = maxy = m_maxIndexY;
	else if (direction.z == 1) z = maxz = m_maxIndexZ;
	else if (direction.x == -1)
	{
		x = 0;
		maxx = 0;
	}
	else if (direction.y == -1)
	{
		y = 0;
		maxy = 0;
	}
	else if (direction.z == -1)
	{
		z = 0;
		maxz = 0;
	}

	for(int i = x ; i<=maxx ; i++)
	{
		for(int j = y ; j<=maxy; j++)
		{
			for(int k = z ; k<=maxz ; k++)
			{
				size_t h = hashValue(i,j,k);
				out.push_back(m_cells[h]);
			}

		}
	}
	return out;


}
*/
template<typename VertexT, typename BoxT>
void HashGrid<VertexT, BoxT>::addLatticePoint(int index_x, int index_y, int index_z, float distance)
{
	size_t hash_value;

	unsigned int INVALID = BoxT::INVALID_INDEX;

	float vsh = 0.5 * this->m_voxelsize;

	// Some iterators for hash map accesses
	typename HashGrid<VertexT, BoxT>::box_map_it it;
	typename HashGrid<VertexT, BoxT>::box_map_it neighbor_it;

	// Values for current and global indices. Current refers to a
	// already present query point, global index is id that the next
	// created query point will get
	unsigned int current_index = 0;

	int dx, dy, dz;

	// Get min and max vertex of the point clouds bounding box
	VertexT v_min = this->m_boundingBox.getMin();
	VertexT v_max = this->m_boundingBox.getMax();

	int e;
	this->m_extrude ? e = 8 : e = 1;
	for(int j = 0; j < e; j++)
	{
		// Get the grid offsets for the neighboring grid position
		// for the given box corner
		dx = HGCreateTable[j][0];
		dy = HGCreateTable[j][1];
		dz = HGCreateTable[j][2];

		hash_value = this->hashValue(index_x + dx, index_y + dy, index_z +dz);

		it = this->m_cells.find(hash_value);
		if(it == this->m_cells.end())
		{
			//Calculate box center
			VertexT box_center(
					(index_x + dx) * this->m_voxelsize + v_min[0],
					(index_y + dy) * this->m_voxelsize + v_min[1],
					(index_z + dz) * this->m_voxelsize + v_min[2]);

			//Create new box
			BoxT* box = new BoxT(box_center);

			//Setup the box itself
			for(int k = 0; k < 8; k++){

				//Find point in Grid
				current_index = this->findQueryPoint(k, index_x + dx, index_y + dy, index_z + dz);
				//If point exist, save index in box
				if(current_index != INVALID) box->setVertex(k, current_index);

					//Otherwise create new grid point and associate it with the current box
				else
				{
					VertexT position(box_center[0] + box_creation_table[k][0] * vsh,
									 box_center[1] + box_creation_table[k][1] * vsh,
									 box_center[2] + box_creation_table[k][2] * vsh);

					this->m_queryPoints.push_back(QueryPoint<VertexT>(position, distance));
					box->setVertex(k, this->m_globalIndex);
					this->m_globalIndex++;

				}
			}

			//Set pointers to the neighbors of the current box
			int neighbor_index = 0;
			size_t neighbor_hash = 0;

			for(int a = -1; a < 2; a++)
			{
				for(int b = -1; b < 2; b++)
				{
					for(int c = -1; c < 2; c++)
					{

						//Calculate hash value for current neighbor cell
						neighbor_hash = this->hashValue(index_x + dx + a,
														index_y + dy + b,
														index_z + dz + c);

						//Try to find this cell in the grid
						neighbor_it = this->m_cells.find(neighbor_hash);

						//If it exists, save pointer in box
						if(neighbor_it != this->m_cells.end())
						{
							box->setNeighbor(neighbor_index, (*neighbor_it).second);
							(*neighbor_it).second->setNeighbor(26 - neighbor_index, box);
						}

						neighbor_index++;
					}
				}
			}

			this->m_cells[hash_value] = box;
		}
	}

}

template<typename VertexT, typename BoxT>
void HashGrid<VertexT, BoxT>::setCoordinateScaling(float x, float y, float z)
{
	m_coordinateScales[0] = x;
	m_coordinateScales[1] = y;
	m_coordinateScales[2] = z;
}

template<typename VertexT, typename BoxT>
HashGrid<VertexT, BoxT>::~HashGrid()
{
	box_map_it iter;
	for(iter = m_cells.begin(); iter != m_cells.end(); iter++)
	{
		if(iter->second != NULL)
		{
			delete (iter->second);
			iter->second = NULL;
		}
	}

	m_cells.clear();
}



template<typename VertexT, typename BoxT>
void HashGrid<VertexT, BoxT>::calcIndices()
{
	float max_size = m_boundingBox.getLongestSide();

	//Save needed grid parameters
	m_maxIndex = (int)ceil( (max_size + 5 * m_voxelsize) / m_voxelsize);
	m_maxIndexSquare = m_maxIndex * m_maxIndex;

	m_maxIndexX = (int)ceil(m_boundingBox.getXSize() / m_voxelsize) + 1;
	m_maxIndexY = (int)ceil(m_boundingBox.getYSize() / m_voxelsize) + 2;
	m_maxIndexZ = (int)ceil(m_boundingBox.getZSize() / m_voxelsize) + 3;
}

template<typename VertexT, typename BoxT>
unsigned int HashGrid<VertexT, BoxT>::findQueryPoint(
		const int &position, const int &x, const int &y, const int &z)
{
	int n_x, n_y, n_z, q_v, offset;
	box_map_it it;

	for(int i = 0; i < 7; i++)
	{
		offset = i * 4;
		n_x = x + shared_vertex_table[position][offset];
		n_y = y + shared_vertex_table[position][offset + 1];
		n_z = z + shared_vertex_table[position][offset + 2];
		q_v = shared_vertex_table[position][offset + 3];

		size_t hash = hashValue(n_x, n_y, n_z);
		//cout << "i=" << i << " looking for hash: " << hash << endl;
		it = m_cells.find(hash);
		if(it != m_cells.end())
		{
		//	cout << "found hash" << endl;
			BoxT* b = it->second;
			if(b->getVertex(q_v) != BoxT::INVALID_INDEX) return b->getVertex(q_v);
		}
		//cout << "did not find hash" << endl;
	}

	return BoxT::INVALID_INDEX;
}


template<typename VertexT, typename BoxT>
void HashGrid<VertexT, BoxT>::saveGrid(string filename)
{
	cout << timestamp << "Writing grid..." << endl;

	// Open file for writing
	ofstream out(filename.c_str());

	// Write data
	if(out.good())
	{
		// Write header
		out << m_queryPoints.size() << " " << m_voxelsize << " " << m_cells.size() << endl;

		// Write query points and distances
		for(size_t i = 0; i < m_queryPoints.size(); i++)
		{
			out << m_queryPoints[i].m_position[0] << " "
					<< m_queryPoints[i].m_position[1] << " "
					<< m_queryPoints[i].m_position[2] << " ";

			if(!isnan(m_queryPoints[i].m_distance))
			{
				out << m_queryPoints[i].m_distance << endl;
			}
			else
			{
				out << 0 << endl;
			}

		}

		// Write box definitions
		typename unordered_map<size_t, BoxT* >::iterator it;
		BoxT* box;
		for(it = m_cells.begin(); it != m_cells.end(); it++)
		{
			box = it->second;
			for(int i = 0; i < 8; i++)
			{
				out << box->getVertex(i) << " ";
			}
			out << endl;
		}
	}
}


template<typename VertexT, typename BoxT>
void HashGrid<VertexT, BoxT>::serialize(string file)
{
	ofstream out(file.c_str());

	// Write data
	if(out.good())
	{
		out <<    m_boundingBox.getMin()[0] << " " << m_boundingBox.getMin()[1]
		<< " " << m_boundingBox.getMin()[2] << " " << m_boundingBox.getMax()[0]
		<< " " << m_boundingBox.getMax()[1] << " " << m_boundingBox.getMax()[2] << endl;


		out << m_queryPoints.size() << " " << m_voxelsize << " " << m_cells.size() << endl;

		// Write query points and distances
		for(size_t i = 0; i < m_queryPoints.size(); i++)
		{
			out << m_queryPoints[i].m_position[0] << " "
			<< m_queryPoints[i].m_position[1] << " "
			<< m_queryPoints[i].m_position[2] << " ";

			if(!isnan(m_queryPoints[i].m_distance))
			{
				out << m_queryPoints[i].m_distance << endl;
			}
			else
			{
				out << 0 << endl;
			}

		}

		// Write box definitions
		typename unordered_map<size_t, BoxT* >::iterator it;
		BoxT* box;
		for(it = m_cells.begin(); it != m_cells.end(); it++)
		{
			box = it->second;
			out << it->first << " ";
			for(int i = 0; i < 8; i++)
			{
				out << box->getVertex(i) << " ";
			}
			out << box->getCenter()[0] << " " << box->getCenter()[1] << " " << box->getCenter()[2] << endl;
		}
	}
	out.close();

	/*
	ofstream ofs(file);
	// write cells amount:
	ofs << m_cells.size() << endl;
	//write cells map
	for(auto it = m_cells.begin() ; it != m_cells.end() ;it++)
	{
		ofs << it->first << " " << it->second;
	}
	ofs << endl;

	//wirte qpIndices size
	ofs << m_qpIndices.size() << endl;
	//write qpIndices map
	for(auto it = m_qpIndices.begin() ; it != m_qpIndices.end() ; it++)
	{
		ofs << it->first << " " << it->second;
	}
	ofs << endl;

	ofs << m_voxelsize << endl;
	ofs << m_maxIndex << endl;
	ofs << m_maxIndexSquare << endl;
	ofs << m_maxIndexX << endl;
	ofs << m_maxIndexY << endl;
	ofs << m_maxIndexZ << endl;

	//write querypoints#
	ofs << m_queryPoints.size() << endl;
	for(auto it = m_queryPoints.begin() ; it != m_queryPoints.end() ; it++)
	{
		ofs << it->m_position[0] << " " << it->m_position[0] << " " <<  it->m_position[0];
		if(!isnan(m_queryPoints[i].m_distance))
		{
			ofs << " " << it->m_distance;
		}
		else
		{
			ofs << " " << 0;
		}
		ofs << endl;
	}

	ofs << m_boxType << endl;
	ofs << m_extrude << endl;
	ofs << m_boundingBox.getMin()[0] << " " << m_boundingBox.getMin()[1] << " " << m_boundingBox.getMin()[2] << endl;
	ofs << m_boundingBox.getMax()[0] << " " << m_boundingBox.getMax()[1] << " " <<  m_boundingBox.getMax()[2] << endl;
	ofs << m_globalIndex << endl;
	ofs << m_coordinateScales[0] << " " << m_coordinateScales[1] << " " << m_coordinateScales[2] << endl;
*/


}

template<typename VertexT, typename BoxT>
Vertex<int> HashGrid<VertexT, BoxT>::convertGlobalToGrid(Vertexf& globalCoord)
{
    Vertex<int> result;
    if(m_extrude)
    {
        result.x = calcIndex((globalCoord[0] - m_boundingBox.getMin()[0] ) / m_voxelsize);
        result.y = calcIndex((globalCoord[1] - m_boundingBox.getMin()[1] ) / m_voxelsize);
        result.z = calcIndex((globalCoord[2] - m_boundingBox.getMin()[2] ) / m_voxelsize);
    }
    else
    {
        result.x = calcIndex((globalCoord[0] - m_boundingBox.getMin()[0]) / m_voxelsize);
        result.y = calcIndex((globalCoord[1] - m_boundingBox.getMin()[1]) / m_voxelsize);
        result.z = calcIndex((globalCoord[2] - m_boundingBox.getMin()[2]) / m_voxelsize);
    }

    return result;
}

} //namespace lvr
