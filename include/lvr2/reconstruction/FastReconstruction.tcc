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
 * FastReconstruction.cpp
 *
 *  Created on: 16.02.2011
 *      Author: Thomas Wiemann
 */
#include "lvr2/geometry/BaseMesh.hpp"
#include "lvr2/reconstruction/FastReconstructionTables.hpp"
#include "lvr2/io/Progress.hpp"

namespace lvr2
{

template<typename BaseVecT, typename BoxT>
FastReconstruction<BaseVecT, BoxT>::FastReconstruction(shared_ptr<HashGrid<BaseVecT, BoxT>> grid)
{
    m_grid = grid;
}

template<typename BaseVecT, typename BoxT>
void FastReconstruction<BaseVecT, BoxT>::getMesh(BaseMesh<BaseVecT> &mesh)
{
    // Status message for mesh generation
    string comment = timestamp.getElapsedTime() + "Creating mesh ";
    ProgressBar progress(m_grid->getNumberOfCells(), comment);

    // Some pointers
    BoxT* b;
    unsigned int global_index = mesh.numVertices();

    // Iterate through cells and calculate local approximations
    typename HashGrid<BaseVecT, BoxT>::box_map_it it;
    for(it = m_grid->firstCell(); it != m_grid->lastCell(); it++)
    {
        b = it->second;
        b->getSurface(mesh, m_grid->getQueryPoints(), global_index);
        if(!timestamp.isQuiet())
            ++progress;
    }

    if(!timestamp.isQuiet())
        cout << endl;

    BoxTraits<BoxT> traits;

    if(traits.type == "SharpBox")  // Perform edge flipping for extended marching cubes
    {
        string SFComment = timestamp.getElapsedTime() + "Flipping edges  ";
        ProgressBar SFProgress(this->m_grid->getNumberOfCells(), SFComment);
        for(it = this->m_grid->firstCell(); it != this->m_grid->lastCell(); it++)
        {

            SharpBox<BaseVecT>* sb;
            sb = reinterpret_cast<SharpBox<BaseVecT>* >(it->second);
            if(sb->m_containsSharpFeature)
            {
                OptionalVertexHandle v1;
                OptionalVertexHandle v2;
                OptionalEdgeHandle e;

                if(sb->m_containsSharpCorner)
                {
                    // 1
                    v1 = sb->m_intersections[ExtendedMCTable[sb->m_extendedMCIndex][0]];
                    v2 = sb->m_intersections[ExtendedMCTable[sb->m_extendedMCIndex][1]];

                    if(v1 && v2)
                    {
                        e = mesh.getEdgeBetween(v1.unwrap(), v2.unwrap());
                        if(e)
                        {
                            mesh.flipEdge(e.unwrap());
                        }
                    }

                    // 2
                    v1 = sb->m_intersections[ExtendedMCTable[sb->m_extendedMCIndex][2]];
                    v2 = sb->m_intersections[ExtendedMCTable[sb->m_extendedMCIndex][3]];

                    if(v1 && v2)
                    {
                        e = mesh.getEdgeBetween(v1.unwrap(), v2.unwrap());
                        if(e)
                        {
                            mesh.flipEdge(e.unwrap());
                        }
                    }

                    // 3
                    v1 = sb->m_intersections[ExtendedMCTable[sb->m_extendedMCIndex][4]];
                    v2 = sb->m_intersections[ExtendedMCTable[sb->m_extendedMCIndex][5]];

                    if(v1 && v2)
                    {
                        e = mesh.getEdgeBetween(v1.unwrap(), v2.unwrap());
                        if(e)
                        {
                            mesh.flipEdge(e.unwrap());
                        }
                    }

                }
                else
                {
                    // 1
                    v1 = sb->m_intersections[ExtendedMCTable[sb->m_extendedMCIndex][0]];
                    v2 = sb->m_intersections[ExtendedMCTable[sb->m_extendedMCIndex][1]];

                    if(v1 && v2)
                    {
                        e = mesh.getEdgeBetween(v1.unwrap(), v2.unwrap());
                        if(e)
                        {
                            mesh.flipEdge(e.unwrap());
                        }
                    }

                    // 2
                    v1 = sb->m_intersections[ExtendedMCTable[sb->m_extendedMCIndex][4]];
                    v2 = sb->m_intersections[ExtendedMCTable[sb->m_extendedMCIndex][5]];

                    if(v1 && v2)
                    {
                        e = mesh.getEdgeBetween(v1.unwrap(), v2.unwrap());
                        if(e)
                        {
                            mesh.flipEdge(e.unwrap());
                        }
                    }
                }
            }
            ++SFProgress;
        }
        cout << endl;
    }

     if(traits.type == "BilinearFastBox")
     {
         string comment = timestamp.getElapsedTime() + "Optimizing plane contours  ";
         ProgressBar progress(this->m_grid->getNumberOfCells(), comment);
         for(it = this->m_grid->firstCell(); it != this->m_grid->lastCell(); it++)
         {
          // F... type safety. According to traits object this is OK!
             BilinearFastBox<BaseVecT>* box = reinterpret_cast<BilinearFastBox<BaseVecT>*>(it->second);
             box->optimizePlanarFaces(mesh, 5);
             ++progress;
         }
         cout << endl;
     }

}

template<typename BaseVecT, typename BoxT>
void FastReconstruction<BaseVecT, BoxT>::getMesh(
    BaseMesh<BaseVecT>& mesh,
    BoundingBox<BaseVecT>& bb,
    vector<unsigned int>& duplicates,
    float comparePrecision
)
{
//    // Status message for mesh generation
//    string comment = timestamp.getElapsedTime() + "Creating Mesh ";
//    ProgressBar progress(m_grid->getNumberOfCells(), comment);

//    // Some pointers
//    BoxT* b;
//    unsigned int global_index = mesh.numVertices();

//    // Iterate through cells and calculate local approximations
//    typename HashGrid<BaseVecT, BoxT>::box_map_it it;
//    for(it = m_grid->firstCell(); it != m_grid->lastCell(); it++)
//    {
//        b = it->second;
//        b->getSurface(mesh, m_grid->getQueryPoints(), global_index, bb, duplicates, comparePrecision);
//        if(!timestamp.isQuiet())
//            ++progress;
//    }

//    if(!timestamp.isQuiet())
//        cout << endl;
}

} // namespace lvr2
