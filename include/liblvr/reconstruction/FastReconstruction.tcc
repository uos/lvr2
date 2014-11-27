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
#include "FastReconstructionTables.hpp"
#include "SharpBox.hpp"
#include "io/Progress.hpp"

namespace lvr
{

template<typename VertexT, typename NormalT, typename BoxT>
FastReconstruction<VertexT, NormalT, BoxT>::FastReconstruction(HashGrid<VertexT, BoxT>* grid)
{
	m_grid = grid;
}

template<typename VertexT, typename NormalT, typename BoxT>
void FastReconstruction<VertexT, NormalT, BoxT>::getMesh(BaseMesh<VertexT, NormalT> &mesh)
{
	// Status message for mesh generation
	string comment = timestamp.getElapsedTime() + "Creating Mesh ";
	ProgressBar progress(m_grid->getNumberOfCells(), comment);

	// Some pointers
	BoxT* b;
	unsigned int global_index = 0;

	// Iterate through cells and calculate local approximations
	typename HashGrid<VertexT, BoxT>::box_map_it it;
	for(it = m_grid->firstCell(); it != m_grid->lastCell(); it++)
	{
		b = it->second;
		b->getSurface(mesh, m_grid->getQueryPoints(), global_index);
		++progress;
	}

	cout << endl;

/*	if(m_boxType == "SF")  // Perform edge flipping for extended marching cubes
	{
		string SFComment = timestamp.getElapsedTime() + "Flipping edges  ";
		ProgressBar SFProgress(m_cells.size(), SFComment);
		for(it = m_cells.begin(); it != m_cells.end(); it++)
		{
			SharpBox<VertexT, typename BoxT, NormalT>* sb;
			sb = (SharpBox<VertexT, typename BoxT, NormalT>*) it->second;
			if(sb->m_containsSharpFeature)
			{
				if(sb->m_containsSharpCorner)
				{
					mesh.flipEdge(sb->m_intersections[ExtendedMCTable[sb->m_extendedMCIndex][0]], sb->m_intersections[ExtendedMCTable[sb->m_extendedMCIndex][1]]);
					mesh.flipEdge(sb->m_intersections[ExtendedMCTable[sb->m_extendedMCIndex][2]], sb->m_intersections[ExtendedMCTable[sb->m_extendedMCIndex][3]]);
					mesh.flipEdge(sb->m_intersections[ExtendedMCTable[sb->m_extendedMCIndex][4]], sb->m_intersections[ExtendedMCTable[sb->m_extendedMCIndex][5]]);
				}
				else
				{
					mesh.flipEdge(sb->m_intersections[ExtendedMCTable[sb->m_extendedMCIndex][0]], sb->m_intersections[ExtendedMCTable[sb->m_extendedMCIndex][1]]);
					mesh.flipEdge(sb->m_intersections[ExtendedMCTable[sb->m_extendedMCIndex][4]], sb->m_intersections[ExtendedMCTable[sb->m_extendedMCIndex][5]]);
				}
			}
			++SFProgress;
		}
		cout << endl;
	
        }

	if(this->m_boxType == "PMC")
	{
	    string comment = timestamp.getElapsedTime() + "Optimizing plane contours  ";
	    ProgressBar progress(m_cells.size(), comment);
	    for(it = m_cells.begin(); it != m_cells.end(); it++)
	    {
	        BilinearFastBox<VertexT, typename BoxT, NormalT>* box = static_cast<BilinearFastBox<VertexT, typename BoxT, NormalT>*> (it->second);
	        box->optimizePlanarFaces(this->m_surface, 5);
	        ++progress;
	    }
	    cout << endl;
	}
*/
}

/*template<typename VertexT, typename NormalT, typename BoxT>
void FastReconstruction<VertexT, typename BoxT, NormalT>::calcQueryPointValues(){

    // Status message output
    string comment = timestamp.getElapsedTime() + "Calculating distance values ";
    ProgressBar progress(m_queryPoints.size(), comment);

    Timestamp ts;

    // Calculate a distance value for each query point
    #pragma omp parallel for
    for( int i = 0; i < (int)m_queryPoints.size(); i++){
        float projectedDistance;
        float euklideanDistance;

        //cout << euklideanDistance << " " << projectedDistance << endl;

        this->m_surface->distance(m_queryPoints[i].m_position, projectedDistance, euklideanDistance);
        if (euklideanDistance > 1.7320 * m_voxelsize)
        {
        	m_queryPoints[i].m_invalid = true;
        }
        m_queryPoints[i].m_distance = projectedDistance;
        ++progress;
    }
    cout << endl;
    cout << timestamp << "Elapsed time: " << ts << endl;
} */

} //namespace lvr
