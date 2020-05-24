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
 * BigGrid.hpp
 *
 *  Created on: Jul 17, 2017
 *      Author: Isaak Mitschke
 */

#ifndef LAS_VEGAS_BIGGRID_HPP
#define LAS_VEGAS_BIGGRID_HPP

#include "lvr2/geometry/BoundingBox.hpp"
#include "lvr2/io/DataStruct.hpp"

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/iostreams/device/mapped_file.hpp>
#include <string>
#include <unordered_map>
#include <utility>

#ifndef __APPLE__
#include <omp.h>
#endif


namespace lvr2
{

using Vec = BaseVector<float>;

struct CellInfo
{
    CellInfo() : size(0), offset(0), inserted(0), dist_offset(0) {}
    size_t size;
    size_t offset;
    size_t inserted;
    size_t dist_offset;
    size_t ix;
    size_t iy;
    size_t iz;
};

template <typename BaseVecT>
class BigGrid
{
  public:
    /**
     * Constructor:
     * @param cloudPath path to PointCloud in ASCII xyz Format // Todo: Add other file formats
     * @param voxelsize
     */
    BigGrid(std::vector<std::string> cloudPath, float voxelsize, float scale = 0, size_t bufferSize = 1024);

    /**
     * Constructor: specific case for incremental reconstruction/chunking. also compatible with simple reconstruction
     * @param voxelsize specified voxelsize
     * @param project ScanProject, which contain one or more Scans
     * @param scale scale value of for current scans
     */
    BigGrid(float voxelsize,ScanProjectEditMarkPtr project, float scale = 0);

    BigGrid(std::string path);

    /**
     * @return Number of voxels
     */
    size_t size();

    /**
     * @return Number Points
     */
    size_t pointSize();

    /**
     * Amount of Points in Voxel at position i,j,k
     * @param i
     * @param j
     * @param k
     * @return amount of points, 0 if voxel does not exsist
     */
    size_t pointSize(int i, int j, int k);

    /**
     * Points of  Voxel at position i,j,k
     * @param i
     * @param j
     * @param k
     * @param numPoints, amount of points in lvr2::floatArr
     * @return lvr2::floatArr, containing points
     */
    lvr2::floatArr points(int i, int j, int k, size_t& numPoints);

    /**
     *  Points that are within bounding box defined by a min and max point
     * @param minx
     * @param miny
     * @param minz
     * @param maxx
     * @param maxy
     * @param maxz
     * @param numPoints number of points
     * @return lvr2::floatArr, containing points
     */
    lvr2::floatArr points(
        float minx, float miny, float minz, float maxx, float maxy, float maxz, size_t& numPoints);

    lvr2::floatArr normals(
        float minx, float miny, float minz, float maxx, float maxy, float maxz, size_t& numPoints);

    lvr2::ucharArr colors(
        float minx, float miny, float minz, float maxx, float maxy, float maxz, size_t& numPoints);
    /**
     * return numbers of points in a specific area (defined by the params) of the grid
     * @param minx
     * @param miny
     * @param minz
     * @param maxx
     * @param maxy
     * @param maxz
     * @return number of points in area
     */
    size_t getSizeofBox(float minx, float miny, float minz, float maxx, float maxy, float maxz);

    void serialize(std::string path = "serinfo.ls");

    lvr2::floatArr getPointCloud(size_t& numPoints);

    BoundingBox<BaseVecT>& getBB() { return m_bb; }

    /**
     *
     * get the partial BB of the area, which needs to be reconstructed
     *
     * @return partial BB
     */
    BoundingBox<BaseVecT>& getpartialBB() { return m_partialbb; }


    virtual ~BigGrid();

    inline size_t hashValue(size_t i, size_t j, size_t k)
    {
        return i * m_maxIndexSquare + j * m_maxIndex + k;
    }

    inline size_t getDistanceFileOffset(size_t hash)
    {
        if (exists(hash))
        {
            return m_gridNumPoints[hash].dist_offset;
        }
        else
            return 0;
    }
    inline bool exists(size_t hash)
    {
        auto it = m_gridNumPoints.find(hash);
        return it != m_gridNumPoints.end();
    }

    inline bool hasColors() { return m_has_color; }
    inline bool hasNormals() { return m_has_normal; }

  private:
    inline int calcIndex(float f) { return f < 0 ? f - .5 : f + .5; }

    bool exists(int i, int j, int k);
    void insert(float x, float y, float z);

    size_t m_maxIndexSquare;
    size_t m_maxIndex;
    size_t m_maxIndexX;
    size_t m_maxIndexY;
    size_t m_maxIndexZ;
    size_t m_numPoints;

    size_t m_pointBufferSize;

    float m_voxelSize;
    bool m_extrude;
#ifdef LVR2_USE_OPEN_MP
    omp_lock_t m_lock;
#endif
    bool m_has_normal;
    bool m_has_color;

    boost::iostreams::mapped_file m_PointFile;
    boost::iostreams::mapped_file m_NomralFile;
    boost::iostreams::mapped_file m_ColorFile;
    BoundingBox<BaseVecT> m_bb;

    //BoundingBox, of unreconstructed scans
    BoundingBox<BaseVecT> m_partialbb;

    std::vector<shared_ptr<Scan>> m_scans;

    std::unordered_map<size_t, CellInfo> m_gridNumPoints;
    float m_scale;
};

} // namespace lvr2

#include "lvr2/reconstruction/BigGrid.tcc"

#endif // LAS_VEGAS_BIGGRID_HPP
