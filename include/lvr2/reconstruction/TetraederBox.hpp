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

namespace lvr2
{

/**
 * @brief Used for Marching Tetreader Reconstruction. Dives a regular box
 *        into 5 Tetraeders for mesh generation.
 */
template<typename BaseVecT>
class TetraederBox : public FastBox<BaseVecT>
{
public:

    /// Creates a new tetraeder box with current grid voxel size
    /// around the given center point
    TetraederBox(Point<BaseVecT> center);
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
        BaseMesh<BaseVecT>& mesh,
        vector<QueryPoint<BaseVecT>>& query_points,
        uint &globalIndex
    );

//    virtual void getSurface(
//        BaseMesh<BaseVecT>& mesh,
//        vector<QueryPoint<BaseVecT>>& query_points,
//        uint& globalIndex,
//        BoundingBox<BaseVecT>& bb,
//        vector<unsigned int>& duplicates,
//        float comparePrecision
//    );

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
            Point<BaseVecT> positions[4],
            float distances[4]
            );

    OptionalVertexHandle    m_intersections[19];
    Point<BaseVecT>         m_intersectionPositionsTetraeder[6];

};

} /* namespace lvr */

#include "TetraederBox.tcc"

#endif /* TETRAEDERBOX_H_ */
