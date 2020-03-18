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
 * HashGrid.hpp
 *
 *  Created on: Nov 27, 2014
 *      Author: twiemann
 */

#ifndef _LVR2_RECONSTRUCTION_HASHGRID_H_
#define _LVR2_RECONSTRUCTION_HASHGRID_H_

#include <unordered_map>
#include <vector>
#include <string>

#include "QueryPoint.hpp"

#include "lvr2/geometry/BoundingBox.hpp"
#include "lvr2/reconstruction/QueryPoint.hpp"

using std::string;
using std::vector;
using std::unordered_map;

namespace lvr2
{

class GridBase
{
public:

    GridBase(bool extrude = true) : m_extrude(extrude) {}

    virtual ~GridBase() {}

    /**
     *
     * @param i         Discrete x position within the grid.
     * @param j         Discrete y position within the grid.
     * @param k         Discrete z position within the grid.
     * @param distance  Signed distance to the represented surface
     *                  at the position within the grid.
     */
    virtual void addLatticePoint(int i, int j, int k, float distance = 0.0) = 0;

    /**
     * @brief   Saves a representation of the grid to the given file
     *
     * @param file      Output file name.
     */
    virtual void saveGrid(string file) = 0;

    /***
     * @brief   Is extrude is set to true, additional cells within the
     *          grid will be create to fill up holes consisting of single
     *          cells.
     *
     * @param extrude   If set to true, additional cells will be created.
     *                  Default value is true.
     */
    virtual void setExtrusion(bool extrude) { m_extrude = extrude; }

protected:
    bool m_extrude;
};

template<typename BaseVecT, typename BoxT>
class HashGrid : public GridBase
{
public:

    BoundingBox<BaseVecT> qp_bb;

    /// Typedef to alias box map
    typedef unordered_map<size_t, BoxT*> box_map;

    typedef unordered_map<size_t, size_t> qp_map;

    /// Typedef to alias iterators for box maps
    typedef typename unordered_map<size_t, BoxT*>::iterator  box_map_it;

    /// Typedef to alias iterators to query points
    typedef typename vector<QueryPoint<BaseVecT>>::iterator query_point_it;

    /***
     * @brief   Constructor
     *
     * If set to true, cell size is interpreted as
     * absolute voxel size (default). Otherwise \ref cellSize
     * is interpreted as number of intersections along the
     * longest size of the given bounding box to estimate a suitable
     * resolution.
     *
     * @param   cellSize        Voxel size of the grid cells
     * @param   isVoxelSize     Whether to interpret \ref cellSize as voxelsize or intersections
     */
    HashGrid(float cellSize, BoundingBox<BaseVecT> boundingBox, bool isVoxelSize = true, bool extrude = true);


    /***
     * @brief   Constructor
     *
     * Construcs a HashGrid from a file
     *
     * @param   file        File representing the HashGrid (See HashGrid::serialize(string file) )
     */
    HashGrid(string file);

    /***
     * @brief Construct a new Hash Grid object
     *
     * @param files
     * @param boundingBox
     * @param voxelsize
     */
    HashGrid(std::vector<string>& files, BoundingBox<BaseVecT>& boundingBox, float voxelsize);


    /***
     * @brief Construct a new Hash Grid object
     *
     * @param files vector of strings to the files which contain the voxel-grid data for the chunks
     * @param innerBoxes vector of BoundingBoxes. Each chunk is only used for the BoundingBox.
     *                          This is important because the data in the chunks may overlap.
     * @param boundingBox bounding box of the complete grid
     * @param voxelsize the voxelsize of the grid
     */
    HashGrid(std::vector<string>& files, std::vector<BoundingBox<BaseVecT>> innerBoxes, BoundingBox<BaseVecT>& boundingBox, float voxelsize);

    /***
     * Constructs a new Hash Grid object from multiple PointBufferPtr,
     * where the HashGrid attributes are saved in the PointBuffer-Channels.
     *
     * @param chunks vector with the voxel-grid data for the chunks
     * @param innerBoxes vector of BoundingBoxes. Each chunk is only used for the BoundingBox.
     *                          This is important because the data in the chunks may overlap.
     * @param boundingBox bounding box of the complete grid
     * @param voxelSize the voxelsize of the grid
     */
    HashGrid(std::vector<PointBufferPtr> chunks,
            std::vector<BoundingBox<BaseVecT>> innerBoxes,
            BoundingBox<BaseVecT>& boundingBox,
            float voxelSize);

    /**
     *
     * @param i         Discrete x position within the grid.
     * @param j         Discrete y position within the grid.
     * @param k         Discrete z position within the grid.
     * @param distance  Signed distance to the represented surface
     *                  at the position within the grid.
     */
    virtual void addLatticePoint(int i, int j, int k, float distance = 0.0);

    /**
     * @brief   Saves a representation of the grid to the given file
     *
     * @param file      Output file name.
     */
    virtual void saveGrid(string file);

    /**
     * @brief Saves a representation of the cells to the given file
     *
     * @param file Output file name.
     */
    void saveCells(string file);

    virtual void serialize(string file);

    /***
     * @brief   Returns the number of generated cells.
     */
    size_t getNumberOfCells() { return m_cells.size(); }

    /**
     * @return  Returns an iterator to the first box in the cell map.
     */
    box_map_it  firstCell() { return m_cells.begin(); }

    /**
     * @return  Returns an iterator to the last box in the cell map.
     */
    box_map_it  lastCell() { return m_cells.end(); }

    /**
     * @return  Returns an iterator to the first query point
     */
    query_point_it firstQueryPoint() { return m_queryPoints.begin(); }

    /***
     * @return  Returns an iterator to the last query point
     */
    query_point_it lastQueryPoint() { return m_queryPoints.end(); }

    vector<QueryPoint<BaseVecT>>& getQueryPoints() { return m_queryPoints; }

    box_map getCells() { return m_cells; }

    /***
     * @brief   Destructor
     */
    virtual ~HashGrid();

    /**
     * @brief   Set x, y, z values to scale the scene or use combinations
     *          of +/-1 to mapp different coordinate systems
     */
    void setCoordinateScaling(float x, float y, float z);

    size_t getMaxIndex() { return m_maxIndex; }

    size_t getMaxIndexX() { return m_maxIndexX; }

    size_t getMaxIndexY() { return m_maxIndexY; }

    size_t getMaxIndexZ() { return m_maxIndexZ; }

    void setBB(BoundingBox<BaseVecT>& bb);

    BoundingBox<BaseVecT> & getBoundingBox() { return m_boundingBox; }

    /**
     * @brief Calculates the hash value for the given index triple
     */
    inline size_t hashValue(int i, int j, int k) const
    {
        return i * m_maxIndexSquare + j * m_maxIndex + k;
    }

    /**
     * @brief   Searches for a existing shared lattice point in the grid.
     *
     * @param position  Number of a possible neighbor
     * @param x         x index within the grid
     * @param y         y index within the grid
     * @param z         z index within the grid
     * @return          Query point index of the found point, INVALID_INDEX otherwise
     */
    unsigned int findQueryPoint(
        int position,
        int x,
        int y,
        int z
    );

    /**
     * @brief   Calculates needed lattice parameters.
     */
    void calcIndices();


protected:

    inline int calcIndex(float f)
    {
        return f < 0 ? f - .5 : f + .5;
    }

    /// Map to handle the boxes in the grid
    box_map         m_cells;

    qp_map          m_qpIndices;

    /// The voxelsize used for reconstruction
    float                       m_voxelsize;

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

    /// A vector containing the query points for the reconstruction
    vector<QueryPoint<BaseVecT>> m_queryPoints;

    /// True if a local tetraeder decomposition is used for reconstruction
    string                      m_boxType;

    /// Bounding box of the covered volume
    BoundingBox<BaseVecT>       m_boundingBox;

    /// The maximum used cell index within the grid
    unsigned int                m_globalIndex;

    /// Save scaling factors (i.e., -1 or +1) to mapp different coordinate systems
    BaseVecT                  m_coordinateScales;
};

} /* namespace lvr */

#include "HashGrid.tcc"

#endif /* _LVR2_RECONSTRUCTION_HASHGRID_H_ */
