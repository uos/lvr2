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
 * BigGrid.cpp
 *
 *  Created on: Jul 17, 2017
 *      Author: Isaak Mitschke
 */

#include "lvr2/io/hdf5/HDF5FeatureBase.hpp"
#include "lvr2/io/LineReader.hpp"
#include "lvr2/io/Progress.hpp"
#include "lvr2/io/Timestamp.hpp"
#include "lvr2/io/hdf5/ArrayIO.hpp"
#include "lvr2/io/hdf5/ChannelIO.hpp"
#include "lvr2/io/hdf5/MatrixIO.hpp"
#include "lvr2/io/hdf5/PointCloudIO.hpp"
#include "lvr2/io/hdf5/VariantChannelIO.hpp"
#include "lvr2/reconstruction/FastReconstructionTables.hpp"

#include <boost/filesystem/path.hpp>
#include <boost/optional/optional_io.hpp>
#include <cstring>
#include <fstream>
#include <iostream>

using namespace std;

namespace lvr2
{

template <typename BaseVecT>
BigGrid<BaseVecT>::BigGrid(std::vector<std::string> cloudPath,
                           float voxelsize,
                           float scale,
                           size_t bufferSize)
    : m_maxIndex(0), m_maxIndexSquare(0), m_maxIndexX(0), m_maxIndexY(0), m_maxIndexZ(0),
      m_numPoints(0), m_extrude(true), m_scale(scale), m_has_normal(false), m_has_color(false),
      m_pointBufferSize(1024)
{

    boost::filesystem::path selectedFile(cloudPath[0]);
    string extension = selectedFile.extension().string();
#ifndef __APPLE__
    omp_init_lock(&m_lock);
#endif
    m_voxelSize = voxelsize;

    // First, parse whole file to get BoundingBox and amount of points
    float ix, iy, iz;
    std::cout << lvr2::timestamp << "Computing Bounding Box..." << std::endl;
    m_numPoints = 0;
    size_t rsize = 0;
    LineReader lineReader(cloudPath);
    size_t lasti = 0;
    while (lineReader.ok())
    {
        if (lineReader.getFileType() == XYZNRGB)
        {
            boost::shared_ptr<xyznc> a = boost::static_pointer_cast<xyznc>(
                lineReader.getNextPoints(rsize, m_pointBufferSize));
            if (rsize <= 0 && !lineReader.ok())
            {
                break;
            }
            for (int i = 0; i < rsize; i++)
            {
                m_bb.expand(BaseVecT(a.get()[i].point.x * m_scale,
                                        a.get()[i].point.y * m_scale,
                                        a.get()[i].point.z * m_scale));
                m_numPoints++;
            }
        }
        else if (lineReader.getFileType() == XYZN)
        {
            boost::shared_ptr<xyzn> a = boost::static_pointer_cast<xyzn>(
                lineReader.getNextPoints(rsize, m_pointBufferSize));
            if (rsize <= 0 && !lineReader.ok())
            {
                break;
            }
            for (int i = 0; i < rsize; i++)
            {
                m_bb.expand(BaseVecT(a.get()[i].point.x * m_scale,
                                        a.get()[i].point.y * m_scale,
                                        a.get()[i].point.z * m_scale));
                m_numPoints++;
            }
        }
        else if (lineReader.getFileType() == XYZ)
        {
            boost::shared_ptr<xyz> a = boost::static_pointer_cast<xyz>(
                lineReader.getNextPoints(rsize, m_pointBufferSize));
            if (rsize <= 0 && !lineReader.ok())
            {
                break;
            }
            for (size_t i = 0; i < rsize; i++)
            {
                m_bb.expand(BaseVecT(a.get()[i].point.x * m_scale,
                                        a.get()[i].point.y * m_scale,
                                        a.get()[i].point.z * m_scale));
                m_numPoints++;
                lasti = i;
            }
        }
        else if (lineReader.getFileType() == XYZRGB)
        {
            boost::shared_ptr<xyzc> a = boost::static_pointer_cast<xyzc>(
                lineReader.getNextPoints(rsize, m_pointBufferSize));
            if (rsize <= 0 && !lineReader.ok())
            {
                break;
            }
            for (size_t i = 0; i < rsize; i++)
            {
                m_bb.expand(BaseVecT(a.get()[i].point.x * m_scale,
                                        a.get()[i].point.y * m_scale,
                                        a.get()[i].point.z * m_scale));
                m_numPoints++;
                lasti = i;
            }
        }
        else
        {
            exit(-1);
        }
    }

    // Make box side lenghts be divisible by voxel size
    float longestSide = m_bb.getLongestSide();

    BaseVecT center = m_bb.getCentroid();
    size_t xsize2 = calcIndex(m_bb.getXSize() / m_voxelSize);
    float xsize = ceil(m_bb.getXSize() / voxelsize) * voxelsize;
    float ysize = ceil(m_bb.getYSize() / voxelsize) * voxelsize;
    float zsize = ceil(m_bb.getZSize() / voxelsize) * voxelsize;
    m_bb.expand(BaseVecT(center.x + xsize / 2, center.y + ysize / 2, center.z + zsize / 2));
    m_bb.expand(BaseVecT(center.x - xsize / 2, center.y - ysize / 2, center.z - zsize / 2));
    longestSide = ceil(longestSide / voxelsize) * voxelsize;

    // calc max indices

    // m_maxIndex = (size_t)(longestSide/voxelsize);
    m_maxIndexX = (size_t)(xsize / voxelsize);
    m_maxIndexY = (size_t)(ysize / voxelsize);
    m_maxIndexZ = (size_t)(zsize / voxelsize);
    m_maxIndex = std::max(m_maxIndexX, std::max(m_maxIndexY, m_maxIndexZ)) + 5 * voxelsize;
    m_maxIndexX += 1;
    m_maxIndexY += 2;
    m_maxIndexZ += 3;
    m_maxIndexSquare = m_maxIndex * m_maxIndex;
    std::cout << "BG: " << m_maxIndexSquare << "|" << m_maxIndexX << "|" << m_maxIndexY << "|"
                << m_maxIndexZ << std::endl;

    string comment = lvr2::timestamp.getElapsedTime() + "Building grid... ";
    lvr2::ProgressBar progress(this->m_numPoints, comment);

    lineReader.rewind();

    size_t idx, idy, idz;
    while (lineReader.ok())
    {
        if (lineReader.getFileType() == XYZNRGB)
        {
            boost::shared_ptr<xyznc> a = boost::static_pointer_cast<xyznc>(
                lineReader.getNextPoints(rsize, m_pointBufferSize));
            if (rsize <= 0 && !lineReader.ok())
            {
                break;
            }
            int dx, dy, dz;
            for (int i = 0; i < rsize; i++)
            {
                ix = a.get()[i].point.x * m_scale;
                iy = a.get()[i].point.y * m_scale;
                iz = a.get()[i].point.z * m_scale;
                idx = calcIndex((ix - m_bb.getMin()[0]) / voxelsize);
                idy = calcIndex((iy - m_bb.getMin()[1]) / voxelsize);
                idz = calcIndex((iz - m_bb.getMin()[2]) / voxelsize);
                int e;
                this->m_extrude ? e = 8 : e = 1;
                for (int j = 0; j < e; j++)
                {
                    dx = HGCreateTable[j][0];
                    dy = HGCreateTable[j][1];
                    dz = HGCreateTable[j][2];
                    size_t h = hashValue(idx + dx, idy + dy, idz + dz);
                    if (j == 0)
                        m_gridNumPoints[h].size++;
                    else
                    {
                        auto it = m_gridNumPoints.find(h);
                        if (it == m_gridNumPoints.end())
                        {
                            m_gridNumPoints[h].size = 0;
                        }
                    }
                }
            }
        }
        else if (lineReader.getFileType() == XYZN)
        {
            boost::shared_ptr<xyzn> a = boost::static_pointer_cast<xyzn>(
                lineReader.getNextPoints(rsize, m_pointBufferSize));
            if (rsize <= 0 && !lineReader.ok())
            {
                break;
            }
            int dx, dy, dz;
            for (int i = 0; i < rsize; i++)
            {
                ix = a.get()[i].point.x * m_scale;
                iy = a.get()[i].point.y * m_scale;
                iz = a.get()[i].point.z * m_scale;
                idx = calcIndex((ix - m_bb.getMin()[0]) / voxelsize);
                idy = calcIndex((iy - m_bb.getMin()[1]) / voxelsize);
                idz = calcIndex((iz - m_bb.getMin()[2]) / voxelsize);

                int e;
                this->m_extrude ? e = 8 : e = 1;
                for (int j = 0; j < e; j++)
                {
                    dx = HGCreateTable[j][0];
                    dy = HGCreateTable[j][1];
                    dz = HGCreateTable[j][2];
                    size_t h = hashValue(idx + dx, idy + dy, idz + dz);
                    if (j == 0)
                        m_gridNumPoints[h].size++;
                    else
                    {
                        auto it = m_gridNumPoints.find(h);
                        if (it == m_gridNumPoints.end())
                        {
                            m_gridNumPoints[h].size = 0;
                        }
                    }
                }
            }
        }
        else if (lineReader.getFileType() == XYZ)
        {
            boost::shared_ptr<xyz> a = boost::static_pointer_cast<xyz>(
                lineReader.getNextPoints(rsize, m_pointBufferSize));
            if (rsize <= 0 && !lineReader.ok())
            {
                break;
            }
            int dx, dy, dz;
            for (int i = 0; i < rsize; i++)
            {
                ix = a.get()[i].point.x * m_scale;
                iy = a.get()[i].point.y * m_scale;
                iz = a.get()[i].point.z * m_scale;
                idx = calcIndex((ix - m_bb.getMin()[0]) / voxelsize);
                idy = calcIndex((iy - m_bb.getMin()[1]) / voxelsize);
                idz = calcIndex((iz - m_bb.getMin()[2]) / voxelsize);
                int e;
                this->m_extrude ? e = 8 : e = 1;
                for (int j = 0; j < e; j++)
                {
                    dx = HGCreateTable[j][0];
                    dy = HGCreateTable[j][1];
                    dz = HGCreateTable[j][2];
                    size_t h = hashValue(idx + dx, idy + dy, idz + dz);
                    if (j == 0)
                        m_gridNumPoints[h].size++;
                    else
                    {
                        auto it = m_gridNumPoints.find(h);
                        if (it == m_gridNumPoints.end())
                        {
                            m_gridNumPoints[h].size = 0;
                        }
                    }
                }
            }
        }
        else if (lineReader.getFileType() == XYZRGB)
        {
            boost::shared_ptr<xyzc> a = boost::static_pointer_cast<xyzc>(
                lineReader.getNextPoints(rsize, m_pointBufferSize));
            if (rsize <= 0 && !lineReader.ok())
            {
                break;
            }
            int dx, dy, dz;
            for (int i = 0; i < rsize; i++)
            {
                ix = a.get()[i].point.x * m_scale;
                iy = a.get()[i].point.y * m_scale;
                iz = a.get()[i].point.z * m_scale;
                idx = calcIndex((ix - m_bb.getMin()[0]) / voxelsize);
                idy = calcIndex((iy - m_bb.getMin()[1]) / voxelsize);
                idz = calcIndex((iz - m_bb.getMin()[2]) / voxelsize);
                int e;
                this->m_extrude ? e = 8 : e = 1;
                for (int j = 0; j < e; j++)
                {
                    dx = HGCreateTable[j][0];
                    dy = HGCreateTable[j][1];
                    dz = HGCreateTable[j][2];
                    size_t h = hashValue(idx + dx, idy + dy, idz + dz);
                    if (j == 0)
                        m_gridNumPoints[h].size++;
                    else
                    {
                        auto it = m_gridNumPoints.find(h);
                        if (it == m_gridNumPoints.end())
                        {
                            m_gridNumPoints[h].size = 0;
                        }
                    }
                }
            }
        }
        else
        {
            exit(-1);
        }
        progress += rsize;
    }

    size_t num_cells = 0;
    size_t offset = 0;
    for (auto it = m_gridNumPoints.begin(); it != m_gridNumPoints.end(); ++it)
    {
        it->second.offset = offset;
        offset += it->second.size;
        it->second.dist_offset = num_cells++;
    }

    lineReader.rewind();

    boost::iostreams::mapped_file_params mmfparam;
    mmfparam.path = "points.mmf";
    mmfparam.mode = std::ios_base::in | std::ios_base::out | std::ios_base::trunc;
    mmfparam.new_file_size = sizeof(float) * m_numPoints * 3;

    boost::iostreams::mapped_file_params mmfparam_normal;
    mmfparam_normal.path = "normals.mmf";
    mmfparam_normal.mode = std::ios_base::in | std::ios_base::out | std::ios_base::trunc;
    mmfparam_normal.new_file_size = sizeof(float) * m_numPoints * 3;

    boost::iostreams::mapped_file_params mmfparam_color;
    mmfparam_color.path = "colors.mmf";
    mmfparam_color.mode = std::ios_base::in | std::ios_base::out | std::ios_base::trunc;
    mmfparam_color.new_file_size = sizeof(unsigned char) * m_numPoints * 3;

    m_PointFile.open(mmfparam);
    float* mmfdata_normal;
    unsigned char* mmfdata_color;
    if (lineReader.getFileType() == XYZNRGB || lineReader.getFileType() == XYZN)
    {
        m_NomralFile.open(mmfparam_normal);
        mmfdata_normal = (float*)m_NomralFile.data();
        m_has_normal = true;
    }
    if (lineReader.getFileType() == XYZNRGB || lineReader.getFileType() == XYZRGB)
    {
        m_ColorFile.open(mmfparam_color);
        mmfdata_color = (unsigned char*)m_ColorFile.data();
        m_has_color = true;
    }
    float* mmfdata = (float*)m_PointFile.data();

    while (lineReader.ok())
    {
        if (lineReader.getFileType() == XYZNRGB)
        {
            boost::shared_ptr<xyznc> a = boost::static_pointer_cast<xyznc>(
                lineReader.getNextPoints(rsize, m_pointBufferSize));
            if (rsize <= 0 && !lineReader.ok())
            {
                break;
            }
            for (int i = 0; i < rsize; i++)
            {
                ix = a.get()[i].point.x * m_scale;
                iy = a.get()[i].point.y * m_scale;
                iz = a.get()[i].point.z * m_scale;
                size_t idx = calcIndex((ix - m_bb.getMin()[0]) / voxelsize);
                size_t idy = calcIndex((iy - m_bb.getMin()[1]) / voxelsize);
                size_t idz = calcIndex((iz - m_bb.getMin()[2]) / voxelsize);
                size_t h = hashValue(idx, idy, idz);
                size_t ins = (m_gridNumPoints[h].inserted);
                m_gridNumPoints[h].ix = idx;
                m_gridNumPoints[h].iy = idy;
                m_gridNumPoints[h].iz = idz;
                m_gridNumPoints[h].inserted++;
                size_t index = m_gridNumPoints[h].offset + ins;
                mmfdata[index * 3] = ix;
                mmfdata[index * 3 + 1] = iy;
                mmfdata[index * 3 + 2] = iz;
                mmfdata_normal[index * 3] = a.get()[i].normal.x;
                mmfdata_normal[index * 3 + 1] = a.get()[i].normal.y;
                mmfdata_normal[index * 3 + 2] = a.get()[i].normal.z;

                mmfdata_color[index * 3] = a.get()[i].color.r;
                mmfdata_color[index * 3 + 1] = a.get()[i].color.g;
                mmfdata_color[index * 3 + 2] = a.get()[i].color.b;
            }
        }
        else if (lineReader.getFileType() == XYZN)
        {
            boost::shared_ptr<xyzn> a = boost::static_pointer_cast<xyzn>(
                lineReader.getNextPoints(rsize, m_pointBufferSize));
            if (rsize <= 0 && !lineReader.ok())
            {
                break;
            }
            for (int i = 0; i < rsize; i++)
            {
                ix = a.get()[i].point.x * m_scale;
                iy = a.get()[i].point.y * m_scale;
                iz = a.get()[i].point.z * m_scale;
                size_t idx = calcIndex((ix - m_bb.getMin()[0]) / voxelsize);
                size_t idy = calcIndex((iy - m_bb.getMin()[1]) / voxelsize);
                size_t idz = calcIndex((iz - m_bb.getMin()[2]) / voxelsize);
                size_t h = hashValue(idx, idy, idz);
                size_t ins = (m_gridNumPoints[h].inserted);
                m_gridNumPoints[h].ix = idx;
                m_gridNumPoints[h].iy = idy;
                m_gridNumPoints[h].iz = idz;
                m_gridNumPoints[h].inserted++;
                size_t index = m_gridNumPoints[h].offset + ins;
                mmfdata[index * 3] = ix;
                mmfdata[index * 3 + 1] = iy;
                mmfdata[index * 3 + 2] = iz;
                mmfdata_normal[index * 3] = a.get()[i].normal.x;
                mmfdata_normal[index * 3 + 1] = a.get()[i].normal.y;
                mmfdata_normal[index * 3 + 2] = a.get()[i].normal.z;
            }
        }
        else if (lineReader.getFileType() == XYZ)
        {
            boost::shared_ptr<xyz> a = boost::static_pointer_cast<xyz>(
                lineReader.getNextPoints(rsize, m_pointBufferSize));
            if (rsize <= 0 && !lineReader.ok())
            {
                break;
            }
            for (int i = 0; i < rsize; i++)
            {
                ix = a.get()[i].point.x * m_scale;
                iy = a.get()[i].point.y * m_scale;
                iz = a.get()[i].point.z * m_scale;
                size_t idx = calcIndex((ix - m_bb.getMin()[0]) / voxelsize);
                size_t idy = calcIndex((iy - m_bb.getMin()[1]) / voxelsize);
                size_t idz = calcIndex((iz - m_bb.getMin()[2]) / voxelsize);
                size_t h = hashValue(idx, idy, idz);
                size_t ins = (m_gridNumPoints[h].inserted);
                m_gridNumPoints[h].ix = idx;
                m_gridNumPoints[h].iy = idy;
                m_gridNumPoints[h].iz = idz;
                m_gridNumPoints[h].inserted++;
                size_t index = m_gridNumPoints[h].offset + ins;
                mmfdata[index * 3] = ix;
                mmfdata[index * 3 + 1] = iy;
                mmfdata[index * 3 + 2] = iz;
            }
        }
        else if (lineReader.getFileType() == XYZRGB)
        {
            boost::shared_ptr<xyzc> a = boost::static_pointer_cast<xyzc>(
                lineReader.getNextPoints(rsize, m_pointBufferSize));
            if (rsize <= 0 && !lineReader.ok())
            {
                break;
            }
            for (int i = 0; i < rsize; i++)
            {
                ix = a.get()[i].point.x * m_scale;
                iy = a.get()[i].point.y * m_scale;
                iz = a.get()[i].point.z * m_scale;
                size_t idx = calcIndex((ix - m_bb.getMin()[0]) / voxelsize);
                size_t idy = calcIndex((iy - m_bb.getMin()[1]) / voxelsize);
                size_t idz = calcIndex((iz - m_bb.getMin()[2]) / voxelsize);
                size_t h = hashValue(idx, idy, idz);
                size_t ins = (m_gridNumPoints[h].inserted);
                m_gridNumPoints[h].ix = idx;
                m_gridNumPoints[h].iy = idy;
                m_gridNumPoints[h].iz = idz;
                m_gridNumPoints[h].inserted++;
                size_t index = m_gridNumPoints[h].offset + ins;
                mmfdata[index * 3] = ix;
                mmfdata[index * 3 + 1] = iy;
                mmfdata[index * 3 + 2] = iz;
                mmfdata_color[index * 3] = a.get()[i].color.r;
                mmfdata_color[index * 3 + 1] = a.get()[i].color.g;
                mmfdata_color[index * 3 + 2] = a.get()[i].color.b;
            }
        }
    }

    //    for(auto it = m_gridNumPoints.begin(); it!= m_gridNumPoints.end() ; it++)
    //    {
    //        std::cout << "h : " << it->first << std::endl;
    //    }
    m_PointFile.close();
    m_NomralFile.close();
    mmfparam.path = "distances.mmf";
    mmfparam.new_file_size = sizeof(float) * size() * 8;

    m_PointFile.open(mmfparam);
    m_PointFile.close();
}


template <typename BaseVecT>
BigGrid<BaseVecT>::BigGrid(float voxelsize, ScanProjectEditMarkPtr project, float scale)
        : m_maxIndex(0), m_maxIndexSquare(0), m_maxIndexX(0), m_maxIndexY(0), m_maxIndexZ(0),
          m_numPoints(0), m_extrude(true), m_scale(scale), m_has_normal(false), m_has_color(false)
{
    /// 
#ifdef LVR2_USE_OPEN_MP
    omp_init_lock(&m_lock);
#endif
    m_voxelSize = voxelsize;

    if (project->changed.size() <= 0)
    {
        std::cerr << "no new scans to be added!" << std::endl;
        return;
    }
    else
    {
        float ix, iy, iz;

        string comment = lvr2::timestamp.getElapsedTime() + "Building grid... ";
        lvr2::ProgressBar progress(project->changed.size() * 3, comment);
        // bounding box of all scans in .h5
        std::vector<BoundingBox<BaseVecT>> scan_boxes;

        //iterate through ALL points to calculate transformed boundingboxes of scans
        for (int i = 0; i < project->changed.size(); i++)
        {
            ScanPositionPtr pos = project->project->positions.at(i); // moegl. Weise project->project ?
            assert(pos->scans.size() > 0);
            size_t numPoints = pos->scans[0]->points->numPoints();
            BoundingBox<BaseVecT> box;

            boost::shared_array<float> points = pos->scans[0]->points->getPointArray();
            Transformd finalPose_n = pos->scans[0]->registration;

            Transformd finalPose = finalPose_n;

            for (int k = 0; k < numPoints; k++)
            {
                Eigen::Vector4d point(
                    points.get()[k * 3], points.get()[k * 3 + 1], points.get()[k * 3 + 2], 1);
                Eigen::Vector4d transPoint = finalPose * point;

                BaseVecT temp(transPoint[0], transPoint[1], transPoint[2]);
                m_bb.expand(temp);
                box.expand(temp);
            }
            // filter the new scans to calculate new reconstruction area
            if(project->changed.at(i))
            {
                m_partialbb.expand(box);
            }
            scan_boxes.push_back(box);

            if(!timestamp.isQuiet())
                ++progress;
        }


        // Make box side lenghts divisible by voxel size
        float longestSide = m_bb.getLongestSide();

        BaseVecT center = m_bb.getCentroid();
        size_t xsize2 = calcIndex(m_bb.getXSize() / m_voxelSize);
        float xsize = ceil(m_bb.getXSize() / voxelsize) * voxelsize;
        float ysize = ceil(m_bb.getYSize() / voxelsize) * voxelsize;
        float zsize = ceil(m_bb.getZSize() / voxelsize) * voxelsize;
        m_bb.expand(BaseVecT(center.x + xsize / 2, center.y + ysize / 2, center.z + zsize / 2));
        m_bb.expand(BaseVecT(center.x - xsize / 2, center.y - ysize / 2, center.z - zsize / 2));
        longestSide = ceil(longestSide / voxelsize) * voxelsize;

        // calculate max indices

        // m_maxIndex = (size_t)(longestSide/voxelsize);
        m_maxIndexX = (size_t)(xsize / voxelsize);
        m_maxIndexY = (size_t)(ysize / voxelsize);
        m_maxIndexZ = (size_t)(zsize / voxelsize);
        m_maxIndex = std::max(m_maxIndexX, std::max(m_maxIndexY, m_maxIndexZ)) + 5 * voxelsize;
        m_maxIndexX += 1;
        m_maxIndexY += 2;
        m_maxIndexZ += 3;
        m_maxIndexSquare = m_maxIndex * m_maxIndex;

        size_t idx, idy, idz;

        for (int i = 0; i < project->changed.size(); i++)
        {
            if ((!project->changed.at(i)) && m_partialbb.isValid() && !m_partialbb.overlap(scan_boxes.at(i)))
            {
                cout << "Scan No. " << i << " ignored!" << endl;
            }
            else
            {

                ScanPositionPtr pos = project->project->positions.at(i);
                size_t numPoints = pos->scans[0]->points->numPoints();
                boost::shared_array<float> points = pos->scans[0]->points->getPointArray();
                m_numPoints += numPoints;
                Transformd finalPose_n = pos->scans[0]->registration;
                Transformd finalPose = finalPose_n;
                int dx, dy, dz;
                for (int k = 0; k < numPoints; k++)
                {
                    Eigen::Vector4d point(
                            points.get()[k * 3], points.get()[k * 3 + 1], points.get()[k * 3 + 2], 1);
                    Eigen::Vector4d transPoint = finalPose * point;
                    BaseVecT temp(transPoint[0], transPoint[1], transPoint[2]);
                    // m_bb.expand(temp);
                    ix = transPoint[0] * m_scale;
                    iy = transPoint[1] * m_scale;
                    iz = transPoint[2] * m_scale;
                    idx = calcIndex((ix - m_bb.getMin()[0]) / voxelsize);
                    idy = calcIndex((iy - m_bb.getMin()[1]) / voxelsize);
                    idz = calcIndex((iz - m_bb.getMin()[2]) / voxelsize);
                    int e;
                    this->m_extrude ? e = 8 : e = 1;
                    for (int j = 0; j < e; j++)
                    {
                        dx = HGCreateTable[j][0];
                        dy = HGCreateTable[j][1];
                        dz = HGCreateTable[j][2];
                        size_t h = hashValue(idx + dx, idy + dy, idz + dz);
                        if (j == 0)
                            m_gridNumPoints[h].size++;
                        else
                        {
                            auto it = m_gridNumPoints.find(h);
                            if (it == m_gridNumPoints.end())
                            {
                                m_gridNumPoints[h].size = 0;

                            }
                        }
                    }
                }
            }
            if(!timestamp.isQuiet())
                ++progress;
        }


        size_t num_cells = 0;
        size_t offset = 0;
        for (auto it = m_gridNumPoints.begin(); it != m_gridNumPoints.end(); ++it)
        {
            it->second.offset = offset;
            offset += it->second.size;
            it->second.dist_offset = num_cells++;
        }

        boost::iostreams::mapped_file_params mmfparam;

        mmfparam.path = "points.mmf";
        mmfparam.mode = std::ios_base::in | std::ios_base::out | std::ios_base::trunc;
        mmfparam.new_file_size = sizeof(float) * m_numPoints * 3;

        boost::iostreams::mapped_file_params mmfparam_normal;
        mmfparam_normal.path = "normals.mmf";
        mmfparam_normal.mode = std::ios_base::in | std::ios_base::out | std::ios_base::trunc;
        mmfparam_normal.new_file_size = sizeof(float) * m_numPoints * 3;

        boost::iostreams::mapped_file_params mmfparam_color;
        mmfparam_color.path = "colors.mmf";
        mmfparam_color.mode = std::ios_base::in | std::ios_base::out | std::ios_base::trunc;
        mmfparam_color.new_file_size = sizeof(unsigned char) * m_numPoints * 3;

        m_PointFile.open(mmfparam);

        float* mmfdata = (float*)m_PointFile.data();



        for (int i = 0; i < project->changed.size(); i++)
        {
            if ((project->changed.at(i) != true) && m_partialbb.isValid() && !m_partialbb.overlap(scan_boxes.at(i)))
            {
                cout << "Scan No. " << i << " ignored!" << endl;
            }
            else{
                ScanPositionPtr pos = project->project->positions.at(i);
                size_t numPoints = pos->scans[0]->points->numPoints();


                boost::shared_array<float> points = pos->scans[0]->points->getPointArray();
                Transformd finalPose_n = pos->scans[0]->registration;
                Transformd finalPose = finalPose_n;
                for (int k = 0; k < numPoints; k++) {
                    Eigen::Vector4d point(
                            points.get()[k * 3], points.get()[k * 3 + 1], points.get()[k * 3 + 2], 1);
                    Eigen::Vector4d transPoint = finalPose * point;

                    ix = transPoint[0] * m_scale;
                    iy = transPoint[1] * m_scale;
                    iz = transPoint[2] * m_scale;
                    size_t idx = calcIndex((ix - m_bb.getMin()[0]) / voxelsize);
                    size_t idy = calcIndex((iy - m_bb.getMin()[1]) / voxelsize);
                    size_t idz = calcIndex((iz - m_bb.getMin()[2]) / voxelsize);
                    size_t h = hashValue(idx, idy, idz);
                    size_t ins = (m_gridNumPoints[h].inserted);
                    m_gridNumPoints[h].ix = idx;
                    m_gridNumPoints[h].iy = idy;
                    m_gridNumPoints[h].iz = idz;
                    m_gridNumPoints[h].inserted++;
                    size_t index = m_gridNumPoints[h].offset + ins;
                    mmfdata[index * 3] = ix;
                    mmfdata[index * 3 + 1] = iy;
                    mmfdata[index * 3 + 2] = iz;
                }
            }
            if(!timestamp.isQuiet())
                ++progress;
        }

        if(!timestamp.isQuiet())
            cout << endl;
        
        m_PointFile.close();
        m_NomralFile.close();
        mmfparam.path = "distances.mmf";
        mmfparam.new_file_size = sizeof(float) * size() * 8;

        m_PointFile.open(mmfparam);
        m_PointFile.close();

    }
}

template <typename BaseVecT>
BigGrid<BaseVecT>::~BigGrid()
{
#ifndef __APPLE__
    omp_destroy_lock(&m_lock);
#endif
}

template <typename BaseVecT>
BigGrid<BaseVecT>::BigGrid(std::string path)
{
    ifstream ifs(path, ios::binary);

    ifs.read((char*)&m_maxIndexSquare, sizeof(m_maxIndexSquare));
    ifs.read((char*)&m_maxIndex, sizeof(m_maxIndex));
    ifs.read((char*)&m_maxIndexX, sizeof(m_maxIndexX));
    ifs.read((char*)&m_maxIndexY, sizeof(m_maxIndexY));
    ifs.read((char*)&m_maxIndexZ, sizeof(m_maxIndexZ));
    ifs.read((char*)&m_numPoints, sizeof(m_numPoints));
    ifs.read((char*)&m_pointBufferSize, sizeof(m_pointBufferSize));
    ifs.read((char*)&m_voxelSize, sizeof(m_voxelSize));
    ifs.read((char*)&m_extrude, sizeof(m_extrude));
    ifs.read((char*)&m_has_normal, sizeof(m_has_normal));
    ifs.read((char*)&m_has_color, sizeof(m_has_color));
    ifs.read((char*)&m_scale, sizeof(m_scale));
    float mx, my, mz, n1, n2, n3;
    ifs.read((char*)&mx, sizeof(float));
    ifs.read((char*)&my, sizeof(float));
    ifs.read((char*)&mz, sizeof(float));
    ifs.read((char*)&n1, sizeof(float));
    ifs.read((char*)&n2, sizeof(float));
    ifs.read((char*)&n3, sizeof(float));
    m_bb.expand(BaseVecT(mx, my, mz));
    m_bb.expand(BaseVecT(n1, n2, n3));

    size_t gridSize;
    ifs.read((char*)&gridSize, sizeof(gridSize));

    std::cout << "LOADING OLD GRID: " << std::endl;
    std::cout << "m_maxIndexSquare: \t\t\t" << m_maxIndexSquare << std::endl;
    std::cout << "m_maxIndex: \t\t\t" << m_maxIndex << std::endl;
    std::cout << "m_maxIndexX: \t\t\t" << m_maxIndexX << std::endl;
    std::cout << "m_maxIndexY: \t\t\t" << m_maxIndexY << std::endl;
    std::cout << "m_maxIndexZ: \t\t\t" << m_maxIndexZ << std::endl;
    std::cout << "m_numPoints: \t\t\t" << m_numPoints << std::endl;
    std::cout << "m_pointBufferSize: \t\t\t" << m_pointBufferSize << std::endl;
    std::cout << "m_voxelSize: \t\t\t" << m_voxelSize << std::endl;
    std::cout << "m_extrude: \t\t\t" << m_extrude << std::endl;
    std::cout << "m_has_normal: \t\t\t" << m_has_normal << std::endl;
    std::cout << "m_scale: \t\t\t" << m_scale << std::endl;
    std::cout << "m_bb: \t\t\t" << m_bb << std::endl;
    std::cout << "gridSize: \t\t\t" << gridSize << std::endl;

    for (size_t i = 0; i < gridSize; i++)
    {
        CellInfo c;
        size_t hash;
        ifs.read((char*)&hash, sizeof(size_t));
        ifs.read((char*)&c.size, sizeof(size_t));
        ifs.read((char*)&c.offset, sizeof(size_t));
        ifs.read((char*)&c.inserted, sizeof(size_t));
        ifs.read((char*)&c.dist_offset, sizeof(size_t));
        ifs.read((char*)&c.ix, sizeof(size_t));
        ifs.read((char*)&c.iy, sizeof(size_t));
        ifs.read((char*)&c.iz, sizeof(size_t));
        m_gridNumPoints[hash] = c;
    }
}

template <typename BaseVecT>
void BigGrid<BaseVecT>::serialize(std::string path)
{
    ofstream ofs(path, ios::binary);
    //    size_t data_size =      sizeof(m_maxIndexSquare) + sizeof(m_maxIndex) +
    //    sizeof(m_maxIndexX) + sizeof(m_maxIndexY) +
    //                            sizeof(m_maxIndexZ) + sizeof(m_numPoints) +
    //                            sizeof(m_pointBufferSize) + sizeof(m_voxelSize) +
    //                            sizeof(m_extrude) + sizeof(m_has_normal) + sizeof(m_has_color) +
    //                            sizeof(m_scale) + (m_gridNumPoints.size() * (sizeof(size_t)*7)) +
    //                            (m_gridNumPoints.size() *(sizeof(size_t)));

    ofs.write((char*)&m_maxIndexSquare, sizeof(m_maxIndexSquare));
    ofs.write((char*)&m_maxIndex, sizeof(m_maxIndex));
    ofs.write((char*)&m_maxIndexX, sizeof(m_maxIndexX));
    ofs.write((char*)&m_maxIndexY, sizeof(m_maxIndexY));
    ofs.write((char*)&m_maxIndexZ, sizeof(m_maxIndexZ));
    ofs.write((char*)&m_numPoints, sizeof(m_numPoints));
    ofs.write((char*)&m_pointBufferSize, sizeof(m_pointBufferSize));
    ofs.write((char*)&m_voxelSize, sizeof(m_voxelSize));
    ofs.write((char*)&m_extrude, sizeof(m_extrude));
    ofs.write((char*)&m_has_normal, sizeof(m_has_normal));
    ofs.write((char*)&m_has_color, sizeof(m_has_color));
    ofs.write((char*)&m_scale, sizeof(m_scale));

    ofs.write((char*)&m_bb.getMin()[0], sizeof(float));
    ofs.write((char*)&m_bb.getMin()[1], sizeof(float));
    ofs.write((char*)&m_bb.getMin()[2], sizeof(float));
    ofs.write((char*)&m_bb.getMax()[0], sizeof(float));
    ofs.write((char*)&m_bb.getMax()[1], sizeof(float));
    ofs.write((char*)&m_bb.getMax()[2], sizeof(float));
    size_t gridSize = m_gridNumPoints.size();
    ofs.write((char*)&gridSize, sizeof(gridSize));
    for (auto it = m_gridNumPoints.begin(); it != m_gridNumPoints.end(); ++it)
    {
        ofs.write((char*)&it->first, sizeof(size_t));
        ofs.write((char*)&it->second.size, sizeof(size_t));
        ofs.write((char*)&it->second.offset, sizeof(size_t));
        ofs.write((char*)&it->second.inserted, sizeof(size_t));
        ofs.write((char*)&it->second.dist_offset, sizeof(size_t));
        ofs.write((char*)&it->second.ix, sizeof(size_t));
        ofs.write((char*)&it->second.iy, sizeof(size_t));
        ofs.write((char*)&it->second.iz, sizeof(size_t));
    }
    ofs.close();
}

template <typename BaseVecT>
size_t BigGrid<BaseVecT>::size()
{
    return m_gridNumPoints.size();
}

template <typename BaseVecT>
size_t BigGrid<BaseVecT>::pointSize()
{
    return m_numPoints;
}

template <typename BaseVecT>
size_t BigGrid<BaseVecT>::pointSize(int i, int j, int k)
{
    size_t h = hashValue(i, j, k);
    auto it = m_gridNumPoints.find(h);
    if (it == m_gridNumPoints.end())
    {
        return 0;
    }
    else
    {
        return m_gridNumPoints[h].size;
    }
}

template <typename BaseVecT>
lvr2::floatArr BigGrid<BaseVecT>::points(int i, int j, int k, size_t& numPoints)
{
    lvr2::floatArr points;
    size_t h = hashValue(i, j, k);
    auto it = m_gridNumPoints.find(h);
    if (it != m_gridNumPoints.end())
    {
        size_t cellSize = m_gridNumPoints[h].size;

        points = lvr2::floatArr(new float[3 * cellSize]);
        boost::iostreams::mapped_file_params mmfparam;
        mmfparam.path = "points.mmf";
        mmfparam.mode = std::ios_base::in | std::ios_base::out | std::ios_base::trunc;

        m_PointFile.open(mmfparam);
        float* mmfdata = (float*)m_PointFile.data();

        memcpy(points.get(), mmfdata, 3 * pointSize() * sizeof(float));

        numPoints = pointSize();
    }
    return points;
}

template <typename BaseVecT>
lvr2::floatArr BigGrid<BaseVecT>::points(
    float minx, float miny, float minz, float maxx, float maxy, float maxz, size_t& numPoints)
{
    minx = (minx > m_bb.getMin()[0]) ? minx : m_bb.getMin()[0];
    miny = (miny > m_bb.getMin()[1]) ? miny : m_bb.getMin()[1];
    minz = (minz > m_bb.getMin()[2]) ? minz : m_bb.getMin()[2];
    maxx = (maxx < m_bb.getMax()[0]) ? maxx : m_bb.getMax()[0];
    maxy = (maxy < m_bb.getMax()[1]) ? maxy : m_bb.getMax()[1];
    maxz = (maxz < m_bb.getMax()[2]) ? maxz : m_bb.getMax()[2];

    size_t idxmin = calcIndex((minx - m_bb.getMin()[0]) / m_voxelSize);
    size_t idymin = calcIndex((miny - m_bb.getMin()[1]) / m_voxelSize);
    size_t idzmin = calcIndex((minz - m_bb.getMin()[2]) / m_voxelSize);
    size_t idxmax = calcIndex((maxx - m_bb.getMin()[0]) / m_voxelSize);
    size_t idymax = calcIndex((maxy - m_bb.getMin()[1]) / m_voxelSize);
    size_t idzmax = calcIndex((maxz - m_bb.getMin()[2]) / m_voxelSize);

    numPoints = getSizeofBox(minx, miny, minz, maxx, maxy, maxz);


    lvr2::floatArr points(new float[numPoints * 3]);
    size_t p_index = 0;

    boost::iostreams::mapped_file_source mmfs("points.mmf");
    float* mmfdata = (float*)mmfs.data();

    for (auto it = m_gridNumPoints.begin(); it != m_gridNumPoints.end(); it++)
    {
        if (it->second.ix >= idxmin && it->second.iy >= idymin && it->second.iz >= idzmin &&
            it->second.ix <= idxmax && it->second.iy <= idymax && it->second.iz <= idzmax)
        {
            size_t cSize = it->second.size;
            for (size_t x = 0; x < cSize; x++)
            {
                points.get()[p_index] = mmfdata[(it->second.offset + x) * 3];
                points.get()[p_index + 1] = mmfdata[(it->second.offset + x) * 3 + 1];
                points.get()[p_index + 2] = mmfdata[(it->second.offset + x) * 3 + 2];
                p_index += 3;
            }
        }
    }
    return points;
}

template <typename BaseVecT>
lvr2::floatArr BigGrid<BaseVecT>::normals(
    float minx, float miny, float minz, float maxx, float maxy, float maxz, size_t& numPoints)
{
    std::ifstream ifs("normals.mmf");
    if (!ifs.good())
    {
        numPoints = 0;
        lvr2::floatArr arr;
        return arr;
    }
    minx = (minx > m_bb.getMin()[0]) ? minx : m_bb.getMin()[0];
    miny = (miny > m_bb.getMin()[1]) ? miny : m_bb.getMin()[1];
    minz = (minz > m_bb.getMin()[2]) ? minz : m_bb.getMin()[2];
    maxx = (maxx < m_bb.getMax()[0]) ? maxx : m_bb.getMax()[0];
    maxy = (maxy < m_bb.getMax()[1]) ? maxy : m_bb.getMax()[1];
    maxz = (maxz < m_bb.getMax()[2]) ? maxz : m_bb.getMax()[2];

    size_t idxmin = calcIndex((minx - m_bb.getMin()[0]) / m_voxelSize);
    size_t idymin = calcIndex((miny - m_bb.getMin()[1]) / m_voxelSize);
    size_t idzmin = calcIndex((minz - m_bb.getMin()[2]) / m_voxelSize);
    size_t idxmax = calcIndex((maxx - m_bb.getMin()[0]) / m_voxelSize);
    size_t idymax = calcIndex((maxy - m_bb.getMin()[1]) / m_voxelSize);
    size_t idzmax = calcIndex((maxz - m_bb.getMin()[2]) / m_voxelSize);

    numPoints = getSizeofBox(minx, miny, minz, maxx, maxy, maxz);

    lvr2::floatArr points(new float[numPoints * 3]);
    size_t p_index = 0;

    boost::iostreams::mapped_file_source mmfs("normals.mmf");
    float* mmfdata = (float*)mmfs.data();

    for (auto it = m_gridNumPoints.begin(); it != m_gridNumPoints.end(); it++)
    {
        if (it->second.ix >= idxmin && it->second.iy >= idymin && it->second.iz >= idzmin &&
            it->second.ix <= idxmax && it->second.iy <= idymax && it->second.iz <= idzmax)
        {
            size_t cSize = it->second.size;
            for (size_t x = 0; x < cSize; x++)
            {
                points.get()[p_index] = mmfdata[(it->second.offset + x) * 3];
                points.get()[p_index + 1] = mmfdata[(it->second.offset + x) * 3 + 1];
                points.get()[p_index + 2] = mmfdata[(it->second.offset + x) * 3 + 2];
                p_index += 3;
            }
        }
    }

    return points;
}

template <typename BaseVecT>
lvr2::ucharArr BigGrid<BaseVecT>::colors(
    float minx, float miny, float minz, float maxx, float maxy, float maxz, size_t& numPoints)
{
    std::ifstream ifs("colors.mmf");
    if (!ifs.good())
    {
        numPoints = 0;
        lvr2::ucharArr arr;
        return arr;
    }
    minx = (minx > m_bb.getMin()[0]) ? minx : m_bb.getMin()[0];
    miny = (miny > m_bb.getMin()[1]) ? miny : m_bb.getMin()[1];
    minz = (minz > m_bb.getMin()[2]) ? minz : m_bb.getMin()[2];
    maxx = (maxx < m_bb.getMax()[0]) ? maxx : m_bb.getMax()[0];
    maxy = (maxy < m_bb.getMax()[1]) ? maxy : m_bb.getMax()[1];
    maxz = (maxz < m_bb.getMax()[2]) ? maxz : m_bb.getMax()[2];

    size_t idxmin = calcIndex((minx - m_bb.getMin()[0]) / m_voxelSize);
    size_t idymin = calcIndex((miny - m_bb.getMin()[1]) / m_voxelSize);
    size_t idzmin = calcIndex((minz - m_bb.getMin()[2]) / m_voxelSize);
    size_t idxmax = calcIndex((maxx - m_bb.getMin()[0]) / m_voxelSize);
    size_t idymax = calcIndex((maxy - m_bb.getMin()[1]) / m_voxelSize);
    size_t idzmax = calcIndex((maxz - m_bb.getMin()[2]) / m_voxelSize);

    numPoints = getSizeofBox(minx, miny, minz, maxx, maxy, maxz);

    lvr2::ucharArr points(new unsigned char[numPoints * 3]);
    size_t p_index = 0;

    boost::iostreams::mapped_file_source mmfs("colors.mmf");
    unsigned char* mmfdata = (unsigned char*)mmfs.data();

    for (auto it = m_gridNumPoints.begin(); it != m_gridNumPoints.end(); it++)
    {
        if (it->second.ix >= idxmin && it->second.iy >= idymin && it->second.iz >= idzmin &&
            it->second.ix <= idxmax && it->second.iy <= idymax && it->second.iz <= idzmax)
        {
            size_t cSize = it->second.size;
            for (size_t x = 0; x < cSize; x++)
            {
                points.get()[p_index] = mmfdata[(it->second.offset + x) * 3];
                points.get()[p_index + 1] = mmfdata[(it->second.offset + x) * 3 + 1];
                points.get()[p_index + 2] = mmfdata[(it->second.offset + x) * 3 + 2];
                p_index += 3;
            }
        }
    }

    return points;
}

template <typename BaseVecT>
bool BigGrid<BaseVecT>::exists(int i, int j, int k)
{
    size_t h = hashValue(i, j, k);
    auto it = m_gridNumPoints.find(h);
    return it != m_gridNumPoints.end();
}

template <typename BaseVecT>
void BigGrid<BaseVecT>::insert(float x, float y, float z)
{
}

template <typename BaseVecT>
lvr2::floatArr BigGrid<BaseVecT>::getPointCloud(size_t& numPoints)
{
    lvr2::floatArr points(new float[3 * pointSize()]);
    boost::iostreams::mapped_file_params mmfparam;
    mmfparam.path = "points.mmf";
    mmfparam.mode = std::ios_base::in | std::ios_base::out | std::ios_base::trunc;

    m_PointFile.open(mmfparam);
    float* mmfdata = (float*)m_PointFile.data();
    memcpy(points.get(), mmfdata, 3 * pointSize() * sizeof(float));

    numPoints = pointSize();
    return points;
}

    template <typename BaseVecT>
size_t BigGrid<BaseVecT>::getSizeofBox(
    float minx, float miny, float minz, float maxx, float maxy, float maxz)
{
    size_t idxmin = calcIndex((minx - m_bb.getMin()[0]) / m_voxelSize);
    size_t idymin = calcIndex((miny - m_bb.getMin()[1]) / m_voxelSize);
    size_t idzmin = calcIndex((minz - m_bb.getMin()[2]) / m_voxelSize);
    size_t idxmax = calcIndex((maxx - m_bb.getMin()[0]) / m_voxelSize);
    size_t idymax = calcIndex((maxy - m_bb.getMin()[1]) / m_voxelSize);
    size_t idzmax = calcIndex((maxz - m_bb.getMin()[2]) / m_voxelSize);

    size_t numPoints = 0;

    // Overhead of saving indices needed to speedup size lookup
    for (auto it = m_gridNumPoints.begin(); it != m_gridNumPoints.end(); it++)
    {
        if (it->second.ix >= idxmin && it->second.iy >= idymin && it->second.iz >= idzmin &&
            it->second.ix <= idxmax && it->second.iy <= idymax && it->second.iz <= idzmax)
        {
            numPoints += it->second.size;
        }
    }

    //    #pragma omp parallel for schedule(dynamic,1) collapse(3)
    //    for(size_t i = idxmin ; i<=idxmax ; i++)
    //    {
    //        for(size_t j = idymin ; j<=idymax ; j++)
    //        {
    //            for(size_t k = idzmin ; k<=idzmax ; k++)
    //            {
    //
    //                size_t h = hashValue(i,j,k);
    //                auto it = m_gridNumPoints.find(h);
    //                if(it != m_gridNumPoints.end())
    //                {
    //                    omp_set_lock(&m_lock);
    //                    numPoints+=it->second.size;
    //                    omp_unset_lock(&m_lock);
    //                }
    //
    //            }
    //        }
    //    }
    return numPoints;
}

} // namespace lvr2
