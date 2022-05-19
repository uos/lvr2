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

#include "lvr2/io/LineReader.hpp"
#include "lvr2/io/baseio/ArrayIO.hpp"
#include "lvr2/io/baseio/MatrixIO.hpp"
#include "lvr2/io/baseio/VariantChannelIO.hpp"
#include "lvr2/io/baseio/ChannelIO.hpp"
#include "lvr2/io/scanio/PointCloudIO.hpp"
#include "lvr2/io/scanio/HDF5IO.hpp"
#include "lvr2/reconstruction/FastReconstructionTables.hpp"
#include "lvr2/util/Progress.hpp"
#include "lvr2/util/Timestamp.hpp"

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
      m_numPoints(0), m_extrude(true), m_scale(scale), m_hasNormal(false), m_hasColor(false),
      m_pointBufferSize(1024)
{

    boost::filesystem::path selectedFile(cloudPath[0]);
    string extension = selectedFile.extension().string();
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

    std::cout << timestamp << "BigGrid - Max Squared indices: " 
              << "\t" << m_maxIndexSquare << "\t " 
              << m_maxIndexX << "\t " 
              << m_maxIndexY << "\t "
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
                calcIndex(BaseVecT(ix, iy, iz), idx, idy, idz);
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
                calcIndex(BaseVecT(ix, iy, iz), idx, idy, idz);

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
                calcIndex(BaseVecT(ix, iy, iz), idx, idy, idz);
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
                calcIndex(BaseVecT(ix, iy, iz), idx, idy, idz);
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
    mmfparam.mode = std::ios_base::in | std::ios_base::out | std::ios_base::trunc;
    mmfparam.new_file_size = sizeof(float) * m_numPoints * 3;

    mmfparam.path = "points.mmf";
    m_PointFile.open(mmfparam);
    float* mmfdata = (float*)m_PointFile.data();

    float* mmfdata_normal;
    unsigned char* mmfdata_color;
    if (lineReader.getFileType() == XYZNRGB || lineReader.getFileType() == XYZN)
    {
        mmfparam.path = "normals.mmf";
        m_NormalFile.open(mmfparam);
        mmfdata_normal = (float*)m_NormalFile.data();
        m_hasNormal = true;
    }
    if (lineReader.getFileType() == XYZNRGB || lineReader.getFileType() == XYZRGB)
    {
        mmfparam.path = "colors.mmf";
        m_ColorFile.open(mmfparam);
        mmfdata_color = (unsigned char*)m_ColorFile.data();
        m_hasColor = true;
    }

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
                size_t idx, idy, idz;
                calcIndex(BaseVecT(ix, iy, iz), idx, idy, idz);
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
                size_t idx, idy, idz;
                calcIndex(BaseVecT(ix, iy, iz), idx, idy, idz);
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
                size_t idx, idy, idz;
                calcIndex(BaseVecT(ix, iy, iz), idx, idy, idz);
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
                size_t idx, idy, idz;
                calcIndex(BaseVecT(ix, iy, iz), idx, idy, idz);
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
}


template <typename BaseVecT>
BigGrid<BaseVecT>::BigGrid(float voxelsize, ScanProjectEditMarkPtr project, float scale)
        : m_maxIndex(0),
          m_maxIndexSquare(0),
          m_maxIndexX(0),
          m_maxIndexY(0),
          m_maxIndexZ(0),
          m_numPoints(0),
          m_extrude(true),
          m_scale(scale),
          m_hasNormal(false),
          m_hasColor(false)
{
    m_voxelSize = voxelsize;

    if (project->changed.size() <= 0)
    {
        std::cout << timestamp << "Warning: No new scans to be added!" << std::endl;
        return;
    }
    else
    {
        float ix, iy, iz;

        string comment = lvr2::timestamp.getElapsedTime() + "Building grid... ";
        lvr2::ProgressBar progress(project->changed.size() * 3, comment);
        
        // Vector of all computed bounding boxes
        std::vector<BoundingBox<BaseVecT>> scan_boxes;


        // Iterate through ALL points to calculate transformed boundingboxes of scans
        for (int i = 0; i < project->changed.size(); i++)
        {
            std::cout << "\r" << timestamp << "Loading scan position " << (i + 1) << " of " << project->changed.size() << std::endl;
            ScanPositionPtr pos = project->project->positions.at(i);
            if(pos && pos->lidars.size())
            {
                // Check if a scan object exists
                LIDARPtr lidar = pos->lidars[0];
                if(lidar->scans.size())
                {
                    // Check if data has already been loaded
                    if(lidar->scans[0] && !lidar->scans[0]->loaded())
                    {
                        lidar->scans[0]->load();
                    }
                    else if(!lidar->scans[0])
                    {
                        // Stored scan has to be a nullptr, try to 
                        // load scan from via scanio
                        auto hdf5io = FeatureBuild<scanio::ScanProjectIO>(project->kernel, project->schema); 
                        std::cout << timestamp << "Overriding empty scan at scan position " << i << std::endl;
                        lidar->scans[0] = hdf5io.ScanIO::load(i, 0, 0);

                        if(!lidar->scans[0])
                        {
                            std::cout << timestamp << "Unable to re-load data. Skipping scan position " << i << std::endl;
                            return;
                        }
                    }
                }
                else
                {
                    std::cout << timestamp << "Loading points from scan position " << i << std::endl;
                    auto hdf5io = FeatureBuild<scanio::ScanProjectIO>(project->kernel, project->schema); 
                    ScanPtr scan = hdf5io.ScanIO::load(i, 0, 0);
                    if(scan)
                    {
                        lidar->scans.push_back(scan);
                    }
                    else
                    {
                        std::cout << timestamp << "Warning: Unable to get data for scan position " << i << std::endl;
                        continue;
                    }
                }
            }

            // Direct acces should be safe now..
            ScanPtr scan = pos->lidars[0]->scans[0];
            
            size_t numPoints = scan->points->numPoints();

            BoundingBox<BaseVecT> box;

            // Get point array
            boost::shared_array<float> points = scan->points->getPointArray();

            // Get transformation from scan position
            Transformd finalPose_n = pos->transformation;

            Transformd finalPose = finalPose_n;

            for (int k = 0; k < numPoints; k++)
            {
                Eigen::Vector4d point(points.get()[k * 3], points.get()[k * 3 + 1], points.get()[k * 3 + 2], 1);
                Eigen::Vector4d transPoint = finalPose * point;

                BaseVecT temp(transPoint[0], transPoint[1], transPoint[2]);
                m_bb.expand(temp);
                box.expand(temp);
            }
            // filter the new scans to calculate new reconstruction area
            if (project->changed.at(i))
            {
                m_partialbb.expand(box);
            }
            scan_boxes.push_back(box);

            if (!timestamp.isQuiet())
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

        // Calculate max indices

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
                cout << timestamp << "Scan No. " << i << " ignored!" << endl;
            }
            else
            {   
                ScanPositionPtr pos = project->project->positions.at(i);
                
                pos->lidars[0]->scans[0]->load();
                size_t numPoints =  pos->lidars[0]->scans[0]->points->numPoints();

                boost::shared_array<float> points = pos->lidars[0]->scans[0]->points->getPointArray();
                m_numPoints += numPoints;
                Transformd finalPose_n = pos->transformation;
                Transformd finalPose = finalPose_n;
                int dx, dy, dz;
                for (int k = 0; k < numPoints; k++)
                {
                    Eigen::Vector4d point(points.get()[k * 3], points.get()[k * 3 + 1], points.get()[k * 3 + 2], 1);
                    Eigen::Vector4d transPoint = finalPose * point;
                    BaseVecT temp(transPoint[0], transPoint[1], transPoint[2]);
                    // m_bb.expand(temp);
                    ix = transPoint[0] * m_scale;
                    iy = transPoint[1] * m_scale;
                    iz = transPoint[2] * m_scale;
                    calcIndex(BaseVecT(ix, iy, iz), idx, idy, idz);
                    int e;
                    this->m_extrude ? e = 8 : e = 1;
                    for (int j = 0; j < e; j++)
                    {
                        dx = HGCreateTable[j][0];
                        dy = HGCreateTable[j][1];
                        dz = HGCreateTable[j][2];
                        size_t h = hashValue(idx + dx, idy + dy, idz + dz);
                        if (j == 0)
                        {
                            m_gridNumPoints[h].size++;
                        }
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
        mmfparam.mode = std::ios_base::in | std::ios_base::out | std::ios_base::trunc;
        mmfparam.new_file_size = sizeof(float) * m_numPoints * 3;

        mmfparam.path = "points.mmf";
        m_PointFile.open(mmfparam);
        float* mmfdata = (float*)m_PointFile.data();

        for (int i = 0; i < project->changed.size(); i++)
        {
            if ((project->changed.at(i) != true) && m_partialbb.isValid() && !m_partialbb.overlap(scan_boxes.at(i)))
            {
                cout << timestamp << "Scan No. " << i << " ignored!" << endl;
            }
            else
            {
                ScanPositionPtr pos = project->project->positions.at(i);
                size_t numPoints = pos->lidars[0]->scans[0]->points->numPoints();


                boost::shared_array<float> points = pos->lidars[0]->scans[0]->points->getPointArray();
                Transformd finalPose_n = pos->transformation;
                Transformd finalPose = finalPose_n;
                for (int k = 0; k < numPoints; k++)
                {
                    Eigen::Vector4d point(
                            points.get()[k * 3], points.get()[k * 3 + 1], points.get()[k * 3 + 2], 1);
                    Eigen::Vector4d transPoint = finalPose * point;

                    ix = transPoint[0] * m_scale;
                    iy = transPoint[1] * m_scale;
                    iz = transPoint[2] * m_scale;
                    size_t idx, idy, idz;
                    calcIndex(BaseVecT(ix, iy, iz), idx, idy, idz);
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
            {
                ++progress;
            }
        }

        if(!timestamp.isQuiet())
        {
            cout << endl;
        }
    }
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
    ifs.read((char*)&m_hasNormal, sizeof(m_hasNormal));
    ifs.read((char*)&m_hasColor, sizeof(m_hasColor));
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

    std::cout << timestamp << "\tLoading Exisiting Grid: " << std::endl;
    std::cout << timestamp << "\tm_maxIndexSquare: \t\t\t" << m_maxIndexSquare << std::endl;
    std::cout << timestamp << "\tm_maxIndex: \t\t\t" << m_maxIndex << std::endl;
    std::cout << timestamp << "\tm_maxIndexX: \t\t\t" << m_maxIndexX << std::endl;
    std::cout << timestamp << "\tm_maxIndexY: \t\t\t" << m_maxIndexY << std::endl;
    std::cout << timestamp << "\tm_maxIndexZ: \t\t\t" << m_maxIndexZ << std::endl;
    std::cout << timestamp << "\tm_numPoints: \t\t\t" << m_numPoints << std::endl;
    std::cout << timestamp << "\tm_pointBufferSize: \t\t\t" << m_pointBufferSize << std::endl;
    std::cout << timestamp << "\tm_voxelSize: \t\t\t" << m_voxelSize << std::endl;
    std::cout << timestamp << "\tm_extrude: \t\t\t" << m_extrude << std::endl;
    std::cout << timestamp << "\tm_hasNormal: \t\t\t" << m_hasNormal << std::endl;
    std::cout << timestamp << "\tm_scale: \t\t\t" << m_scale << std::endl;
    std::cout << timestamp << "\tm_bb: \t\t\t" << m_bb << std::endl;
    std::cout << timestamp << "\tGridSize: \t\t\t" << gridSize << std::endl;

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

    boost::iostreams::mapped_file_params mmfparam;
    mmfparam.mode = std::ios_base::in | std::ios_base::out | std::ios_base::trunc;
    mmfparam.new_file_size = sizeof(float) * m_numPoints * 3;

    mmfparam.path = "points.mmf";
    m_PointFile.open(mmfparam);

    if (m_hasNormal)
    {
        mmfparam.path = "normals.mmf";
        m_NormalFile.open(mmfparam);
    }
    if (m_hasColor)
    {
        mmfparam.path = "colors.mmf";
        m_ColorFile.open(mmfparam);
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
    //                            sizeof(m_extrude) + sizeof(m_hasNormal) + sizeof(m_hasColor) +
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
    ofs.write((char*)&m_hasNormal, sizeof(m_hasNormal));
    ofs.write((char*)&m_hasColor, sizeof(m_hasColor));
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
    return it != m_gridNumPoints.end() ? it->second.size : 0;
}

template <typename BaseVecT>
lvr2::floatArr BigGrid<BaseVecT>::points(int i, int j, int k, size_t& numPoints)
{
    lvr2::floatArr points;
    size_t h = hashValue(i, j, k);
    auto it = m_gridNumPoints.find(h);
    if (it != m_gridNumPoints.end())
    {
        auto& cell = it->second;

        points = lvr2::floatArr(new float[3 * cell.size]);

        float* cellData = (float*)m_PointFile.data() + 3 * cell.offset;

        std::copy(cellData, cellData + 3 * cell.size, points.get());

        numPoints = cell.size;
    }
    return points;
}

template <typename BaseVecT>
lvr2::floatArr BigGrid<BaseVecT>::points(BaseVecT min, BaseVecT max, size_t& numPoints)
{
    min.x = std::max(min.x, m_bb.getMin().x);
    min.y = std::max(min.y, m_bb.getMin().y);
    min.z = std::max(min.z, m_bb.getMin().z);
    max.x = std::min(max.x, m_bb.getMax().x);
    max.y = std::min(max.y, m_bb.getMax().y);
    max.z = std::min(max.z, m_bb.getMax().z);

    size_t idxmin, idymin, idzmin, idxmax, idymax, idzmax;
    calcIndex(min, idxmin, idymin, idzmin);
    calcIndex(max, idxmax, idymax, idzmax);

    std::vector<std::pair<size_t, size_t>> cellCounts;
    numPoints = getSizeofBox(min, max, cellCounts);

    lvr2::floatArr points(new float[numPoints * 3]);

    // determine where each cell is going to start in the point array
    std::vector<float*> cellOutPoints;
    cellOutPoints.push_back(points.get());
    for (auto& [ id, count ] : cellCounts)
    {
        cellOutPoints.push_back(cellOutPoints.back() + count * 3);
    }

    float* pointFile = (float*)m_PointFile.data();

    #pragma omp parallel for
    for (size_t i = 0; i < cellCounts.size(); i++)
    {
        auto& [ id, cellNumPoints ] = cellCounts[i];
        auto& cell = m_gridNumPoints[id];

        float* cellOut = cellOutPoints[i];

        float* cellIn = pointFile + 3 * cell.offset;
        float* cellInEnd = cellIn + 3 * cell.size;
        for (float* p = cellIn; p < cellInEnd; p += 3)
        {
            if (p[0] >= min.x && p[0] <= max.x && p[1] >= min.y && p[1] <= max.y && p[2] >= min.z && p[2] <= max.z)
            {
                cellOut = std::copy(p, p + 3, cellOut);
            }
        }

        size_t n = (cellOut - cellOutPoints[i]) / 3;
        if (n != cellNumPoints)
        {
            throw std::runtime_error("BigGrid::points: inconsistent number of points");
        }
    }

    return points;
}

template <typename BaseVecT>
lvr2::floatArr BigGrid<BaseVecT>::normals(BaseVecT min, BaseVecT max, size_t& numNormals)
{
    if (!m_hasNormal)
    {
        numNormals = 0;
        return lvr2::floatArr();
    }
    min.x = std::max(min.x, m_bb.getMin().x);
    min.y = std::max(min.y, m_bb.getMin().y);
    min.z = std::max(min.z, m_bb.getMin().z);
    max.x = std::min(max.x, m_bb.getMax().x);
    max.y = std::min(max.y, m_bb.getMax().y);
    max.z = std::min(max.z, m_bb.getMax().z);

    size_t idxmin, idymin, idzmin, idxmax, idymax, idzmax;
    calcIndex(min, idxmin, idymin, idzmin);
    calcIndex(max, idxmax, idymax, idzmax);

    std::vector<std::pair<size_t, size_t>> cellCounts;
    numNormals = getSizeofBox(min, max, cellCounts);

    lvr2::floatArr normals(new float[numNormals * 3]);

    // determine where each cell is going to start in the point array
    std::vector<float*> cellOutNormals;
    cellOutNormals.push_back(normals.get());
    for (auto& [ id, count ] : cellCounts)
    {
        cellOutNormals.push_back(cellOutNormals.back() + count * 3);
    }

    float* pointFile = (float*)m_PointFile.data();
    float* normalFile = (float*)m_NormalFile.data();

    #pragma omp parallel for
    for (size_t i = 0; i < cellCounts.size(); i++)
    {
        auto& [ id, cellNumNormals ] = cellCounts[i];
        auto& cell = m_gridNumPoints[id];

        float* cellOut = cellOutNormals[i];

        float* cellIn = normalFile + 3 * cell.offset;
        float* points = pointFile + 3 * cell.offset;
        float* pointsEnd = points + 3 * cell.size;
        for (float* p = points; p < pointsEnd; p += 3, cellIn += 3)
        {
            if (p[0] >= min.x && p[0] <= max.x && p[1] >= min.y && p[1] <= max.y && p[2] >= min.z && p[2] <= max.z)
            {
                cellOut = std::copy(cellIn, cellIn + 3, cellOut);
            }
        }

        size_t n = (cellOut - cellOutNormals[i]) / 3;
        if (n != cellNumNormals)
        {
            throw std::runtime_error("BigGrid::normals: inconsistent number of normals");
        }
    }

    return normals;
}

template <typename BaseVecT>
lvr2::ucharArr BigGrid<BaseVecT>::colors(BaseVecT min, BaseVecT max, size_t& numColors)
{
    if (!m_hasColor)
    {
        numColors = 0;
        return lvr2::ucharArr();
    }
    min.x = std::max(min.x, m_bb.getMin().x);
    min.y = std::max(min.y, m_bb.getMin().y);
    min.z = std::max(min.z, m_bb.getMin().z);
    max.x = std::min(max.x, m_bb.getMax().x);
    max.y = std::min(max.y, m_bb.getMax().y);
    max.z = std::min(max.z, m_bb.getMax().z);

    size_t idxmin, idymin, idzmin, idxmax, idymax, idzmax;
    calcIndex(min, idxmin, idymin, idzmin);
    calcIndex(max, idxmax, idymax, idzmax);

    std::vector<std::pair<size_t, size_t>> cellCounts;
    numColors = getSizeofBox(min, max, cellCounts);

    lvr2::ucharArr colors(new float[numColors * 3]);

    // determine where each cell is going to start in the point array
    std::vector<uchar*> cellOutColors;
    cellOutColors.push_back(colors.get());
    for (auto& [ id, count ] : cellCounts)
    {
        cellOutColors.push_back(cellOutColors.back() + count * 3);
    }

    float* pointFile = (float*)m_PointFile.data();
    uchar* colorFile = (uchar*)m_ColorFile.data();

    #pragma omp parallel for
    for (size_t i = 0; i < cellCounts.size(); i++)
    {
        auto& [ id, cellNumColors ] = cellCounts[i];
        auto& cell = m_gridNumPoints[id];

        uchar* cellOut = cellOutColors[i];

        uchar* cellIn = colorFile + 3 * cell.offset;
        float* points = pointFile + 3 * cell.offset;
        float* pointsEnd = points + 3 * cell.size;
        for (float* p = points; p < pointsEnd; p += 3, cellIn += 3)
        {
            if (p[0] >= min.x && p[0] <= max.x && p[1] >= min.y && p[1] <= max.y && p[2] >= min.z && p[2] <= max.z)
            {
                cellOut = std::copy(cellIn, cellIn + 3, cellOut);
            }
        }

        size_t n = (cellOut - cellOutColors[i]) / 3;
        if (n != cellNumColors)
        {
            throw std::runtime_error("BigGrid::colors: inconsistent number of colors");
        }
    }

    return colors;
}

template <typename BaseVecT>
bool BigGrid<BaseVecT>::exists(int i, int j, int k)
{
    size_t h = hashValue(i, j, k);
    auto it = m_gridNumPoints.find(h);
    return it != m_gridNumPoints.end();
}

template <typename BaseVecT>
lvr2::floatArr BigGrid<BaseVecT>::getPointCloud(size_t& numPoints)
{
    numPoints = m_numPoints;

    lvr2::floatArr points(new float[3 * numPoints]);

    float* pointData = (float*)m_PointFile.data();
    std::copy(pointData, pointData + 3 * numPoints, points.get());

    return points;
}

template <typename BaseVecT>
size_t BigGrid<BaseVecT>::getSizeofBox(BaseVecT min, BaseVecT max, std::vector<std::pair<size_t, size_t>>& cellCounts)
{
    size_t idxmin, idymin, idzmin, idxmax, idymax, idzmax;
    calcIndex(min, idxmin, idymin, idzmin);
    calcIndex(max, idxmax, idymax, idzmax);

    cellCounts.clear();

    for (auto& [ id, cell ] : m_gridNumPoints)
    {
        if (cell.ix >= idxmin || cell.iy >= idymin || cell.iz >= idzmin ||
            cell.ix < idxmax || cell.iy < idymax || cell.iz < idzmax)
        {
            cellCounts.emplace_back(id, 0);
        }
    }

    size_t numPoints = 0;
    float* pointFile = (float*)m_PointFile.data();

    #pragma omp parallel for reduction(+:numPoints)
    for (size_t i = 0; i < cellCounts.size(); i++)
    {
        auto& [ id, cellNumPoints ] = cellCounts[i];
        auto& cell = m_gridNumPoints[id];

        float* cellPoints = pointFile + 3 * cell.offset;
        float* cellPointsEnd = cellPoints + 3 * cell.size;
        for (float* p = cellPoints; p < cellPointsEnd; p += 3)
        {
            if (p[0] >= min.x && p[0] <= max.x && p[1] >= min.y && p[1] <= max.y && p[2] >= min.z && p[2] <= max.z)
            {
                cellNumPoints++;
                numPoints++;
            }
        }
    }

    return numPoints;
}

} // namespace lvr2
