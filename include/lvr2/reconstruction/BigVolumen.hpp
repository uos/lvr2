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
 * BigVolumen.hpp
 *
 *  Created on: Jul 17, 2017
 *      Author: Isaak Mitschke
 */

#ifndef LAS_VEGAS_BigVolumen_HPP
#define LAS_VEGAS_BigVolumen_HPP

#include "lvr2/geometry/BoundingBox.hpp"
#include "lvr2/io/DataStruct.hpp"

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/iostreams/device/mapped_file.hpp>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>

#ifndef __APPLE__
#include <omp.h>
#endif

namespace lvr2
{

template <typename BaseVecT>
class BigVolumen
{

  public:
    struct VolumeCellInfo
    {
        VolumeCellInfo() : size(0), overlapping_size(0) {}
        size_t size;
        size_t overlapping_size;
        size_t ix;
        size_t iy;
        size_t iz;
        std::string path;
        BoundingBox<BaseVecT> bb;
        std::ofstream ofs_points;
        std::ofstream ofs_normals;
        std::ofstream ofs_colors;
    };

    /**
     * Constructor:
     * @param cloudPath path to PointCloud in ASCII xyz Format // Todo: Add other file formats
     * @param voxelsize
     */
    BigVolumen(std::vector<std::string>,
               float voxelsize,
               float overlapping_size = 0,
               float scale = 1);

    /**
     * @return Number of voxels
     */
    size_t size();

    /**
     * @return Number Points
     */
    size_t pointSize();

    BoundingBox<BaseVecT>& getBB() { return m_bb; }

    virtual ~BigVolumen();

    inline size_t hashValue(size_t i, size_t j, size_t k)
    {
        return i * m_maxIndexSquare + j * m_maxIndex + k;
    }

    inline bool hasColors() { return m_has_color; }
    inline bool hasNormals() { return m_has_normal; }

    inline std::unordered_map<size_t, VolumeCellInfo>* getCellinfo() { return &m_gridNumPoints; };

  private:
    inline int calcIndex(float f) { return f < 0 ? f - .5 : f + .5; }

    size_t m_maxIndexSquare;
    size_t m_maxIndex;
    size_t m_maxIndexX;
    size_t m_maxIndexY;
    size_t m_maxIndexZ;
    size_t m_numPoints;
    float m_voxelSize;
    bool m_extrude;

    bool m_has_normal;
    bool m_has_color;

    boost::iostreams::mapped_file m_PointFile;
    boost::iostreams::mapped_file m_NomralFile;
    boost::iostreams::mapped_file m_ColorFile;
    BoundingBox<BaseVecT> m_bb;
    std::unordered_map<size_t, VolumeCellInfo> m_gridNumPoints;
    float m_scale;
};

} // namespace lvr2

#include "lvr2/reconstruction/BigVolumen.tcc"

#endif // LAS_VEGAS_BigVolumen_HPP
