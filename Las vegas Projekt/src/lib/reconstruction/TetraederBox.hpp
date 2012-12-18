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
 * TetraederBox.h
 *
 *  @date 23.11.2011
 *  @author Thomas Wiemann
 */

#ifndef TETRAEDERBOX_H_
#define TETRAEDERBOX_H_


#include "FastBox.hpp"

namespace lssr
{

/**
 * @brief Used for Marching Tetreader Reconstruction. Dives a regular box
 *        into 5 Tetraeders for mesh generation.
 */
template<typename VertexT, typename NormalT>
class TetraederBox : public FastBox<VertexT, NormalT>
{
public:
    TetraederBox(VertexT);
    virtual ~TetraederBox();

    /**
     * @brief Performs a local reconstruction using a tetraeder decomposition
     *        of the current cell
     *
     * @param mesh          The reconstructed mesh
     * @param query_points  A vector containing the query points of the
     *                      reconstruction grid
     * @param globalIndex   The index of the newest vertex in the mesh, i.e.
     *                      a newly generated vertex shout have the index
     *                      globalIndex + 1.
     */
    virtual void getSurface(
            BaseMesh<VertexT, NormalT> &mesh,
            vector<QueryPoint<VertexT> > &query_points,
            uint &globalIndex);


private:

    int calcPatternIndex(float distances[4])
    {
        int index = 0;
        for(int i = 0; i < 4; i++)
        {
            if(distances[i] > 0) index |= (1 << i);
        }
        return index;
    }

    inline void interpolateIntersections(
            int tetraNumber,
            VertexT positions[4],
            float distances[4]
            );

    uint     m_intersections[19];
    VertexT  m_intersectionPositionsTetraeder[6];

};

} /* namespace lssr */

#include "TetraederBox.tcc"

#endif /* TETRAEDERBOX_H_ */
