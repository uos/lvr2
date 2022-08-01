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

#include "lvr2/types/PointBuffer.hpp"
#include "lvr2/types/MatrixTypes.hpp"


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
    virtual void saveGrid(std::string file) = 0;

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
    /// type alias for box map
    typedef std::unordered_map<Vector3i, BoxT*> box_map;

    /// type alias for list of query points
    typedef std::vector<QueryPoint<BaseVecT>*> qp_list;

    /// Typedef to alias iterators for box maps
    typedef typename box_map::iterator box_map_it;

    /// Typedef to alias iterators to query points
    typedef typename qp_list::iterator query_point_it;

    /**
     * @brief Construct an empty HashGrid object.
     *
     * @param resolution if isVoxelsize: voxel size. if not: number of voxels on the longest size of bb.
     * @param bb the bounding box of the grid
     * @param isVoxelsize see resolution description
     * @param extrude add cells around the existing ones
     */
    HashGrid(float resolution, BoundingBox<BaseVecT> bb, bool isVoxelsize = true, bool extrude = true);

    /**
     * @brief Construct a new Hash Grid object from a file.
     * 
     * @param file A file written by saveGrid(file).
     * @param boundingBox Bounding box of the grid.
     * @param voxelsize The voxel size of the grid.
     */
    HashGrid(std::string file, const BoundingBox<BaseVecT>& boundingBox, float voxelsize)
        : HashGrid({ file }, boundingBox, voxelsize)
    { }

    /**
     * @brief Construct a new Hash Grid object from multiple files.
     * 
     * @param files A list of files each written by saveGrid(file).
     * @param boundingBox Bounding box of the grid.
     * @param voxelsize The voxel size of the grid.
     */
    HashGrid(const std::vector<std::string>& files, const BoundingBox<BaseVecT>& boundingBox, float voxelsize)
        : HashGrid(files, {}, boundingBox, voxelsize)
    { }

    /**
     * @brief Construct a new Hash Grid object from multiple files.
     * 
     * @param files A list of files each written by saveGrid(file).
     * @param innerBoxes A list of bounding boxes for each file. Useful for removing overlap.
     * @param boundingBox Bounding box of the grid.
     * @param voxelsize The voxel size of the grid.
     */
    HashGrid(const std::vector<string>& files,
             const std::vector<BoundingBox<BaseVecT>>& innerBoxes,
             const BoundingBox<BaseVecT>& boundingBox,
             float voxelsize);

    /**
     * @brief Construct a new Hash Grid object from a PointBuffer.
     * 
     * @param src A PointBuffer created with toPointBuffer().
     * @param boundingBox Bounding box of the grid.
     * @param voxelsize The voxel size of the grid.
     */
    HashGrid(PointBufferPtr src, const BoundingBox<BaseVecT>& boundingBox, float voxelsize)
        : HashGrid({ src }, {}, boundingBox, voxelsize)
    { }

    /**
     * @brief Construct a new Hash Grid object from multiple PointBuffers.
     * 
     * @param chunks A list of PointBuffers each created with toPointBuffer().
     * @param innerBoxes A list of bounding boxes for each buffer. Useful for removing overlap.
     * @param boundingBox Bounding box of the grid.
     * @param voxelsize The voxel size of the grid.
     */
    HashGrid(const std::vector<PointBufferPtr>& chunks,
             const std::vector<BoundingBox<BaseVecT>>& innerBoxes,
             const BoundingBox<BaseVecT>& boundingBox,
             float voxelsize);

    /**
     * @brief Create a PointBuffer containing the cells and distances.
     *
     * Can be converted back to a HashGrid with HashGrid(PointBufferPtr, ...).
     */
    PointBufferPtr toPointBuffer() const;

    /**
     * @brief Add one cell to the grid. When adding multiple cells, use addLatticePoints()
     *
     * Note that distances are measured at the *Corners* of a cell, and the distance parameter
     * is only used to fill remaining corners that aren't shared with existing cells.
     *
     * @param i         Discrete x position within the grid.
     * @param j         Discrete y position within the grid.
     * @param k         Discrete z position within the grid.
     * @param distance  Signed distance to the represented surface
     *                  at the position within the grid.
     */
    void addLatticePoint(int i, int j, int k, float distance = 0.0) override;

    /**
     * @brief Add many lattice points at once
     * 
     * @param indices the {i,j,k} indices of the lattice points
     */
    void addLatticePoints(const std::unordered_set<Vector3i>& indices);

    /**
     * @brief Saves a representation of the grid to the given file
     *
     * @param file Output file name.
     */
    void saveGrid(std::string file) override;

    /***
     * @brief   Returns the number of generated cells.
     */
    size_t getNumberOfCells() const { return m_cells.size(); }

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

    std::vector<QueryPoint<BaseVecT>>& getQueryPoints() { return m_queryPoints; }
    const std::vector<QueryPoint<BaseVecT>>& getQueryPoints() const { return m_queryPoints; }

    box_map& getCells() { return m_cells; }
    const box_map& getCells() const { return m_cells; }

    /***
     * @brief   Destructor
     */
    virtual ~HashGrid();

    void setBB(BoundingBox<BaseVecT>& bb) { m_boundingBox = bb; }

    BoundingBox<BaseVecT>& getBoundingBox() { return m_boundingBox; }
    const BoundingBox<BaseVecT>& getBoundingBox() const { return m_boundingBox; }

    /**
     * @brief   Searches for a existing shared lattice point in the grid.
     *
     * @param position  Number of a possible neighbor
     * @param index     index within the grid
     * @return          Query point index of the found point, INVALID_INDEX otherwise
     */
    uint findQueryPoint(int position, const Vector3i& index) const;

    void calcIndex(const BaseVecT& vec, Vector3i& index) const
    {
        index.x() = std::floor(vec.x / m_voxelsize);
        index.y() = std::floor(vec.y / m_voxelsize);
        index.z() = std::floor(vec.z / m_voxelsize);
    }
    Vector3i calcIndex(const BaseVecT& vec) const
    {
        Vector3i ret;
        calcIndex(vec, ret);
        return ret;
    }

    void indexToCenter(const Vector3i& index, BaseVecT& center) const
    {
        center.x = (index.x() + 0.5f) * m_voxelsize;
        center.y = (index.y() + 0.5f) * m_voxelsize;
        center.z = (index.z() + 0.5f) * m_voxelsize;
    }
    BaseVecT indexToCenter(const Vector3i& index) const
    {
        BaseVecT ret;
        indexToCenter(index, ret);
        return ret;
    }

protected:
    /// Fill in the neighbors of all cells
    void fillNeighbors();

    BoxT* addBox(const Vector3i& index, const BaseVecT& center, float* distances);
    BoxT* addBox(const BaseVecT& center, float* distances)
    {
        return addBox(calcIndex(center), center, distances);
    }
    BoxT* addBox(const Vector3i& index, float* distances)
    {
        return addBox(index, indexToCenter(index), distances);
    }

    /// Map to handle the boxes in the grid
    box_map m_cells;

    /// The voxelsize used for reconstruction
    float m_voxelsize;

    /// A vector containing the query points for the reconstruction
    std::vector<QueryPoint<BaseVecT>> m_queryPoints;

    /// Bounding box of the covered volume
    BoundingBox<BaseVecT> m_boundingBox;
};

} /* namespace lvr */

#include "HashGrid.tcc"

#endif /* _LVR2_RECONSTRUCTION_HASHGRID_H_ */
