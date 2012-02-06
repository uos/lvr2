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
 * FastReconstruction.h
 *
 *  Created on: 16.02.2011
 *      Author: Thomas Wiemann
 */

#ifndef FastReconstruction_H_
#define FastReconstruction_H_

#include "geometry/Vertex.hpp"
#include "geometry/BoundingBox.hpp"
#include "reconstruction/PointsetMeshGenerator.hpp"
#include "reconstruction/LocalApproximation.hpp"
#include "reconstruction/FastBox.hpp"
//#include "reconstruction/PlanarFastBox.hpp"
#include "reconstruction/TetraederBox.hpp"
#include "reconstruction/SharpBox.hpp"
#include "reconstruction/QueryPoint.hpp"
#include "reconstruction/PointsetSurface.hpp"

#include <ext/hash_map>
using namespace __gnu_cxx;

namespace lssr
{

/**
 * @brief A surface reconstruction object that implements the standard
 *        marching cubes algorithm using a hashed grid structure for
 *        parallel computation.
 */
template<typename VertexT, typename NormalT>
class FastReconstruction : public PointsetMeshGenerator<VertexT, NormalT>
{
public:

    /**
     * @brief Constructor.
     *
     * @param manager       A point cloud manager instance
     * @param resolution    The number of intersections (on the longest side
     *                      of the volume taken by the data points) used by
     *                      the reconstruction.
     * @param isVoxelsize   If set to true, interpret resolution as voxelsize
     *                      instead of number of intersections
     */
    FastReconstruction( 
            typename PointsetSurface<VertexT>::Ptr surface,
            float resolution,
            bool isVoxelsize = false,
            string boxType = "MC");


    /**
     * @brief Destructor.
     */
    virtual ~FastReconstruction() {};

    /**
     * @brief Returns the surface reconstruction of the given point set.
     *
     * @param mesh
     */
    virtual void getMesh(BaseMesh<VertexT, NormalT> &mesh);

    /**
     * @brief Saves a grid representation to an ASCII file. File format is as follows:
     *        First line declares the number of query points in the grid and the cells size. Then one point
     *        per line is defined (x, y, z, distance). After that, tuples of eight
     *        indices per line define the grid cells.
     */
    void saveGrid(string filename);

private:

    /**
     * @brief Calculates the maximal grid indices
     */
    void calcIndices();

    /**
     * @brief Creates the needed query points for the reconstruction
     *        processs
     */
    void createGrid();

    /**
     * @brief Calculates the distance for all generated query points
     */
    void calcQueryPointValues();

    /**
     * @brief Tries to find an existing query point in the grid for
     *        the virtual box corner (1..8) for the  cell at (i, j, k)
     *        in the grid.
     *
     * @param position  The box corner index
     * @param i, j, k   A triple that identifies a cell in the grid
     *
     * @return The index of an existing query point or -1 if no point
     *         corresponding to the given position exists.
     */
    uint findQueryPoint(const int &position,
            const int &i, const int &j, const int &k);

    /**
     * @brief Calculates the hash value for the given index tripel
     */
    inline size_t hashValue(int i, int j, int k) const
    {
        return i * m_maxIndexSquare + j * m_maxIndex + k;
    }

    /**
     * @brief Rounds the given value to the neares integer value
     */
    inline int calcIndex(float f)
    {
        return f < 0 ? f-.5:f+.5;
    }

    /// The voxelsize used for reconstruction
    float                  		m_voxelsize;

    /// The absolute maximal index of the reconstruction grid
    size_t                      m_maxIndex;

    /// The squared maximal index of the reconstruction grid
    size_t                      m_maxIndexSquare;

    /// The maximal index in x direction
    size_t                      m_maxIndexX;

    /// The maximal index in y direction
    size_t                      m_maxIndexY;

    /// The maximal index in z direction
    size_t                      m_maxIndexZ;

    /// A hahs map to store the created grid cells
    hash_map<size_t, FastBox<VertexT, NormalT>* >  m_cells;

    /// A vector containing the query points for the reconstruction
    vector<QueryPoint<VertexT> > m_queryPoints;

    /// True if a local tetraeder decomposition is used for reconstruction
    string                        m_boxType;

};


} // namespace lssr


#include "FastReconstruction.tcc"

#endif /* FastReconstruction_H_ */
