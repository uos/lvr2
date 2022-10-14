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

/**
 * BigGrid.tcc
 *
 * @date Jul 17, 2017
 * @author Isaak Mitschke
 * @author Malte Hillmann
 */

#pragma once

#include "lvr2/io/LineReader.hpp"
#include "lvr2/io/scanio/HDF5IO.hpp"
#include "lvr2/util/Progress.hpp"
#include "lvr2/util/Timestamp.hpp"

#include <cstring>
#include <fstream>
#include <iostream>

namespace lvr2
{

template <typename BaseVecT>
BigGrid<BaseVecT>::BigGrid(std::vector<std::string> cloudPath,
                           float voxelsize,
                           float scale,
                           bool extrude,
                           size_t bufferSize)
    : m_numPoints(0), m_extrude(extrude), m_scale(scale), m_hasNormal(false), m_hasColor(false),
      m_pointBufferSize(1024)
{

    fs::path selectedFile(cloudPath[0]);
    std::string extension = selectedFile.extension().string();
    m_voxelSize = voxelsize;

    // First, parse whole file to get BoundingBox and amount of points
    float ix, iy, iz;
    std::cout << lvr2::timestamp << "[BigGrid] Computing Bounding Box..." << std::endl;
    m_numPoints = 0;

    LineReader lineReader(cloudPath);

    if (lineReader.getFileType() == XYZNRGB)
    {
        initFromLineReader<xyznc>(lineReader);
    }
    else if (lineReader.getFileType() == XYZN)
    {
        initFromLineReader<xyzn>(lineReader);
    }
    else if (lineReader.getFileType() == XYZ)
    {
        initFromLineReader<xyz>(lineReader);
    }
    else if (lineReader.getFileType() == XYZRGB)
    {
        initFromLineReader<xyzc>(lineReader);
    }
    else
    {
        throw std::runtime_error("Unsupported LineReader type");
    }

    if (m_extrude)
    {
        calcExtrusion();
    }
}

template<typename BaseVecT>
template<typename LineType>
void BigGrid<BaseVecT>::initFromLineReader(LineReader& lineReader)
{
    size_t rsize = 0;
    Vector3i index;

    while (lineReader.ok())
    {
        auto a = boost::static_pointer_cast<LineType>(lineReader.getNextPoints(rsize, m_pointBufferSize));
        if (rsize <= 0 && !lineReader.ok())
        {
            break;
        }
        for (size_t i = 0; i < rsize; i++)
        {
            auto& in = a.get()[i].point;
            auto point = BaseVecT(in.x, in.y, in.z) * m_scale;
            m_bb.expand(point);
            m_numPoints++;

            calcIndex(BaseVecT(point.x, point.y, point.z) * m_scale, index);

            getCellInfo(index).size++;
        }
    }

    // Make box side lengths be divisible by voxel size
    BaseVecT center = m_bb.getCentroid();
    float xsize = ceil(m_bb.getXSize() / m_voxelSize) * m_voxelSize;
    float ysize = ceil(m_bb.getYSize() / m_voxelSize) * m_voxelSize;
    float zsize = ceil(m_bb.getZSize() / m_voxelSize) * m_voxelSize;
    m_bb.expand(BaseVecT(center.x + xsize / 2, center.y + ysize / 2, center.z + zsize / 2));
    m_bb.expand(BaseVecT(center.x - xsize / 2, center.y - ysize / 2, center.z - zsize / 2));

    size_t offset = 0;

    for (auto& [ index, cell ] : m_cells)
    {
        cell.offset = offset;
        offset += cell.size;
        m_numPoints += cell.size;
    }

    boost::iostreams::mapped_file_params mmfparam;
    mmfparam.mode = std::ios_base::in | std::ios_base::out | std::ios_base::trunc;
    mmfparam.new_file_size = sizeof(float) * m_numPoints * 3;

    mmfparam.path = (m_pathPrefix / "points.mmf").string();
    m_PointFile.open(mmfparam);
    float* mmfdata = (float*)m_PointFile.data();

    float* mmfdata_normal;
    unsigned char* mmfdata_color;
    if constexpr(LineTypeTraits<LineType>::hasNormal)
    {
        mmfparam.path = (m_pathPrefix / "normals.mmf").string();
        m_NormalFile.open(mmfparam);
        mmfdata_normal = (float*)m_NormalFile.data();
        m_hasNormal = true;
    }
    if constexpr(LineTypeTraits<LineType>::hasColor)
    {
        mmfparam.path = (m_pathPrefix / "colors.mmf").string();
        m_ColorFile.open(mmfparam);
        mmfdata_color = (unsigned char*)m_ColorFile.data();
        m_hasColor = true;
    }

    lineReader.rewind();

    while (lineReader.ok())
    {
        auto a = boost::static_pointer_cast<LineType>(lineReader.getNextPoints(rsize, m_pointBufferSize));
        if (rsize <= 0 && !lineReader.ok())
        {
            break;
        }
        for (size_t i = 0; i < rsize; i++)
        {
            auto& in = a.get()[i];
            auto point = BaseVecT(in.point.x, in.point.y, in.point.z) * m_scale;
            auto& cell = getCellInfo(point);
            size_t pos = (cell.offset + cell.inserted) * 3;
            cell.inserted++;
            mmfdata[pos] = point.x;
            mmfdata[pos + 1] = point.y;
            mmfdata[pos + 2] = point.z;
            if constexpr(LineTypeTraits<LineType>::hasNormal)
            {
                mmfdata_normal[pos] = in.normal.x;
                mmfdata_normal[pos + 1] = in.normal.y;
                mmfdata_normal[pos + 2] = in.normal.z;
            }
            if constexpr(LineTypeTraits<LineType>::hasColor)
            {
                mmfdata_color[pos] = in.color.r;
                mmfdata_color[pos + 1] = in.color.g;
                mmfdata_color[pos + 2] = in.color.b;
            }
        }
    }
}


template <typename BaseVecT>
BigGrid<BaseVecT>::BigGrid(float voxelsize, ScanProjectEditMarkPtr project, const fs::path& pathPrefix, float scale, bool extrude)
        : m_numPoints(0),
          m_voxelSize(voxelsize),
          m_extrude(extrude),
          m_scale(scale),
          m_pathPrefix(pathPrefix),
          m_hasNormal(false),
          m_hasColor(false)
{
    if (project->changed.size() <= 0)
    {
        std::cout << timestamp << "[BigGrid] Warning: No new scans to be added!" << std::endl;
        return;
    }

    size_t numScans = project->changed.size();

    // Vector of all computed bounding boxes
    std::vector<BoundingBox<BaseVecT>> scanBoxes(numScans);
    std::vector<std::unordered_map<Vector3i, CellInfo>> scanCells(numScans);
    std::vector<bool> ignoredOrInvalid(numScans, false);

    std::stringstream ss;
    ss << timestamp << "[BigGrid] Building grid: loading " << numScans << " scan positions";
    lvr2::ProgressBar progressLoading(numScans, ss.str());

    // Iterate through ALL points to calculate transformed boundingboxes of scans
    for (size_t i = 0; i < numScans; i++)
    {
        ScanPositionPtr pos = project->project->positions.at(i);
        if (!pos || pos->lidars.empty())
        {
            std::cout << timestamp << "[BigGrid] Warning: scan position " << i << " is empty" << std::endl;
            ignoredOrInvalid[i] = true;
            ++progressLoading;
            continue;
        }
        // Check if a scan object exists
        LIDARPtr lidar = pos->lidars[0];
        if(lidar->scans.empty() || !lidar->scans[0])
        {
            std::cout << timestamp << "[BigGrid] Loading points with scanio" << std::endl;
            auto hdf5io = scanio::HDF5IOBase(project->kernel, project->schema); 
            ScanPtr scan = hdf5io.ScanIO::load(i, 0, 0);
            if(!scan)
            {
                std::cout << timestamp << "[BigGrid] Warning: Unable to get data for scan position " << i << std::endl;
                ignoredOrInvalid[i] = true;
                ++progressLoading;
                continue;
            }
            if (lidar->scans.empty())
            {
                lidar->scans.push_back(scan);
            }
            else
            {
                lidar->scans[0] = scan;
            }
        }
        ScanPtr scan = lidar->scans[0];
        bool wasLoaded = scan->loaded();
        scan->load();

        size_t numPoints = scan->points->numPoints();
        boost::shared_array<float> points = scan->points->getPointArray();

        // Get transformation from scan position
        Transformd finalPose = pos->transformation;

        auto& box = scanBoxes[i];
        auto& scanCell = scanCells[i];

        #pragma omp parallel
        {
            BoundingBox<BaseVecT> local_bb;
            std::unordered_map<Vector3i, CellInfo> local_cells;

            #pragma omp for schedule(static) nowait
            for (size_t k = 0; k < numPoints; k++)
            {
                Eigen::Vector4d original(points.get()[k * 3], points.get()[k * 3 + 1], points.get()[k * 3 + 2], 1);
                Eigen::Vector4d transPoint = finalPose * original;

                auto point = BaseVecT(transPoint[0], transPoint[1], transPoint[2]) * m_scale;
                local_bb.expand(point);

                Vector3i index = calcIndex(point);
                local_cells[index].size++;
            }
            #pragma omp critical
            {
                box.expand(local_bb);
                m_bb.expand(local_bb);

                for (auto& [ index, cell ] : local_cells)
                {
                    scanCell[index].size += cell.size;
                }
            }
        }
        // filter the new scans to calculate new reconstruction area
        if (project->changed[i])
        {
            m_partialbb.expand(box);
        }

        // unload scan to save memory. Scans that were already loaded might not have a way of
        // loading them again, so keep them loaded for now
        if (!wasLoaded)
        {
            scan->release();
        }

        ++progressLoading;
    }
    std::cout << std::endl;

    for (size_t i = 0; i < numScans; i++)
    {
        if (ignoredOrInvalid[i])
        {
            continue;
        }
        if (!project->changed[i] && m_partialbb.isValid() && !m_partialbb.overlap(scanBoxes[i]))
        {
            std::cout << timestamp << "[BigGrid] Scan No. " << i << " ignored!" << std::endl;
            ignoredOrInvalid[i] = true;
            continue;
        }
        for (auto& [ index, cell ] : scanCells[i])
        {
            getCellInfo(index).size += cell.size;
        }
    }
    scanCells.clear();

    // Make box side lengths divisible by voxel size
    BaseVecT center = m_bb.getCentroid();
    float xsize = ceil(m_bb.getXSize() / voxelsize) * voxelsize;
    float ysize = ceil(m_bb.getYSize() / voxelsize) * voxelsize;
    float zsize = ceil(m_bb.getZSize() / voxelsize) * voxelsize;
    m_bb.expand(BaseVecT(center.x + xsize / 2, center.y + ysize / 2, center.z + zsize / 2));
    m_bb.expand(BaseVecT(center.x - xsize / 2, center.y - ysize / 2, center.z - zsize / 2));

    size_t offset = 0;

    auto it = m_cells.begin();
    size_t erased = 0, oldSize = m_cells.size();
    while (it != m_cells.end())
    {
        auto& cell = it->second;
        if (cell.size < 20)
        {
            // ~1/3 of all cells tend to have abysmally few points in them, which make everything
            // slower while providing very little benefit. Therefore, we remove them.
            erased += cell.size;
            it = m_cells.erase(it);
        }
        else
        {
            cell.offset = offset;
            offset += cell.size;
            m_numPoints += cell.size;

            ++it;
        }
    }
    if (erased > 0)
    {
        std::cout << timestamp << "[BigGrid] Removed " << erased << " points from "
                  << (oldSize - m_cells.size()) << " tiny cells." << std::endl;
        std::cout << timestamp << "[BigGrid] " << m_numPoints << " in " << m_cells.size() << " cells remaining" << std::endl;
    }

    boost::iostreams::mapped_file_params mmfparam;
    mmfparam.mode = std::ios_base::in | std::ios_base::out | std::ios_base::trunc;
    mmfparam.new_file_size = sizeof(float) * m_numPoints * 3;

    mmfparam.path = (m_pathPrefix / "points.mmf").string();
    m_PointFile.open(mmfparam);
    float* mmfdata = (float*)m_PointFile.data();

    ss.str("");
    ss << timestamp << "[BigGrid] Building grid: filling cells";
    lvr2::ProgressBar progressFilling(numScans, ss.str());

    Eigen::Vector4d original, transPoint;
    BaseVecT point;
    Vector3i index;

    for (size_t i = 0; i < numScans; i++)
    {
        if (ignoredOrInvalid[i])
        {
            ++progressFilling;
            continue;
        }
        ScanPositionPtr pos = project->project->positions.at(i);
        Transformd finalPose = pos->transformation;

        ScanPtr scan = pos->lidars[0]->scans[0];
        scan->load();
        size_t numPoints = scan->points->numPoints();
        boost::shared_array<float> points = scan->points->getPointArray();

        for (size_t k = 0; k < numPoints; k++)
        {
            original << points.get()[k * 3], points.get()[k * 3 + 1], points.get()[k * 3 + 2], 1;
            transPoint = finalPose * original;

            point = BaseVecT(transPoint[0], transPoint[1], transPoint[2]) * m_scale;
            calcIndex(point, index);
            auto it = m_cells.find(index);
            if (it == m_cells.end())
            {
                continue;
            }
            auto& cell = it->second;
            size_t pos = (cell.offset + cell.inserted) * 3;
            cell.inserted++;
            mmfdata[pos] = point.x;
            mmfdata[pos + 1] = point.y;
            mmfdata[pos + 2] = point.z;
        }
        scan->release();

        ++progressFilling;
    }
    std::cout << std::endl;

    if (m_extrude)
    {
        calcExtrusion();
    }

    // consistency check
    bool failed = false;
    for (auto& [ index, cell ] : m_cells)
    {
        if (cell.inserted != cell.size)
        {
            std::cout << timestamp << "[BigGrid] Cell " << index.transpose() << ": " << cell.inserted << "/" << cell.size << std::endl;
            failed = true;
        }
    }
    if (failed)
    {
        throw std::runtime_error("BigGrid creation failed: Inconsistent number of points in cells");
    }
}

template<typename T>
void fread(std::ifstream& file, T& value)
{
    file.read((char*)value, sizeof(T));
}

template <typename BaseVecT>
BigGrid<BaseVecT>::BigGrid(std::string path)
{
    ifstream ifs(path, ios::binary);

    fread(ifs, m_numPoints);
    fread(ifs, m_pointBufferSize);
    fread(ifs, m_voxelSize);
    fread(ifs, m_extrude);
    fread(ifs, m_hasNormal);
    fread(ifs, m_hasColor);
    fread(ifs, m_scale);
    BaseVecT min, max;
    fread(ifs, min.x); fread(ifs, min.y); fread(ifs, min.z);
    fread(ifs, max.x); fread(ifs, max.y); fread(ifs, max.z);
    m_bb = BoundingBox<BaseVecT>(min, max);

    size_t gridSize;
    fread(ifs, gridSize);

    std::cout << timestamp << "[BigGrid] \tLoading Exisiting Grid: " << std::endl;
    std::cout << timestamp << "[BigGrid] \tm_numPoints: \t\t\t" << m_numPoints << std::endl;
    std::cout << timestamp << "[BigGrid] \tm_pointBufferSize: \t\t\t" << m_pointBufferSize << std::endl;
    std::cout << timestamp << "[BigGrid] \tm_voxelSize: \t\t\t" << m_voxelSize << std::endl;
    std::cout << timestamp << "[BigGrid] \tm_extrude: \t\t\t" << m_extrude << std::endl;
    std::cout << timestamp << "[BigGrid] \tm_hasNormal: \t\t\t" << m_hasNormal << std::endl;
    std::cout << timestamp << "[BigGrid] \tm_scale: \t\t\t" << m_scale << std::endl;
    std::cout << timestamp << "[BigGrid] \tm_bb: \t\t\t" << m_bb << std::endl;
    std::cout << timestamp << "[BigGrid] \tGridSize: \t\t\t" << gridSize << std::endl;

    for (size_t i = 0; i < gridSize; i++)
    {
        Vector3i index;
        fread(ifs, index.x()); fread(ifs, index.y()); fread(ifs, index.z());

        auto& c = getCellInfo(index);
        fread(ifs, c.size);
        fread(ifs, c.offset);
        fread(ifs, c.inserted);
    }

    boost::iostreams::mapped_file_params mmfparam;
    mmfparam.mode = std::ios_base::in | std::ios_base::out | std::ios_base::trunc;
    mmfparam.new_file_size = sizeof(float) * m_numPoints * 3;

    mmfparam.path = (m_pathPrefix / "points.mmf").string();
    m_PointFile.open(mmfparam);

    if (m_hasNormal)
    {
        mmfparam.path = (m_pathPrefix / "normals.mmf").string();
        m_NormalFile.open(mmfparam);
    }
    if (m_hasColor)
    {
        mmfparam.path = (m_pathPrefix / "colors.mmf").string();
        m_ColorFile.open(mmfparam);
    }
}

template<typename T>
void fwrite(std::ofstream& file, T& value)
{
    file.write((char*)value, sizeof(T));
}

template <typename BaseVecT>
void BigGrid<BaseVecT>::serialize(std::string path)
{
    ofstream ofs(path, ios::binary);
    fwrite(ofs, m_numPoints);
    fwrite(ofs, m_pointBufferSize);
    fwrite(ofs, m_voxelSize);
    fwrite(ofs, m_extrude);
    fwrite(ofs, m_hasNormal);
    fwrite(ofs, m_hasColor);
    fwrite(ofs, m_scale);

    BaseVecT min = m_bb.min();
    BaseVecT max = m_bb.max();
    fwrite(ofs, min.x); fwrite(ofs, min.y); fwrite(ofs, min.z);
    fwrite(ofs, max.x); fwrite(ofs, max.y); fwrite(ofs, max.z);

    size_t gridSize = m_cells.size();
    fwrite(ofs, gridSize);

    for (auto& [ index, cell ] : m_cells)
    {
        fwrite(ofs, index.x()); fwrite(ofs, index.y()); fwrite(ofs, index.z());

        fwrite(ofs, cell.size);
        fwrite(ofs, cell.offset);
        fwrite(ofs, cell.inserted);
    }
    ofs.close();
}

template <typename BaseVecT>
BigGrid<BaseVecT>::~BigGrid()
{
    m_PointFile.close();
    fs::remove(m_pathPrefix / "points.mmf");

    if (m_hasNormal)
    {
        m_NormalFile.close();
        fs::remove(m_pathPrefix / "normals.mmf");
    }
    if (m_hasColor)
    {
        m_ColorFile.close();
        fs::remove(m_pathPrefix / "colors.mmf");
    }
}

template <typename BaseVecT>
void BigGrid<BaseVecT>::calcExtrusion()
{
    std::unordered_set<Vector3i> newCells;
    for (auto& [ index, _ ] : m_cells)
    {
        for (int dx = -1; dx <= 1; dx++)
        {
            for (int dy = -1; dy <= 1; dy++)
            {
                for (int dz = -1; dz <= 1; dz++)
                {
                    Vector3i di = index + Vector3i(dx, dy, dz);
                    if (m_cells.find(di) == m_cells.end())
                    {
                        newCells.insert(di);
                    }
                }
            }
        }
    }
    for (auto& index : newCells)
    {
        m_cells[index];
    }
}

template <typename BaseVecT>
size_t BigGrid<BaseVecT>::size()
{
    return m_cells.size();
}

template <typename BaseVecT>
size_t BigGrid<BaseVecT>::pointSize()
{
    return m_numPoints;
}

template <typename BaseVecT>
size_t BigGrid<BaseVecT>::pointSize(const Vector3i& index)
{
    auto it = m_cells.find(index);
    return it != m_cells.end() ? it->second.size : 0;
}

template <typename BaseVecT>
lvr2::floatArr BigGrid<BaseVecT>::points(const Vector3i& index, size_t& numPoints) const
{
    lvr2::floatArr points;
    auto it = m_cells.find(index);
    if (it != m_cells.end() && it->second.size > 0)
    {
        auto& cell = it->second;

        points = lvr2::floatArr(new float[3 * cell.size]);

        const float* cellData = (const float*)m_PointFile.data() + 3 * cell.offset;

        std::copy_n(cellData, 3 * cell.size, points.get());

        numPoints = cell.size;
    }
    return points;
}

template <typename BaseVecT>
lvr2::floatArr BigGrid<BaseVecT>::points(const BoundingBox<BaseVecT>& bb, size_t& numPoints, size_t minNumPoints) const
{
    std::vector<std::pair<const CellInfo*, size_t>> cellCounts;
    numPoints = getSizeofBox(bb, cellCounts);

    if (numPoints < minNumPoints)
    {
        return lvr2::floatArr();
    }

    lvr2::floatArr points(new float[numPoints * 3]);

    // determine where each cell is going to start in the point array
    std::vector<float*> cellOutPoints;
    cellOutPoints.push_back(points.get());
    for (auto& [ cell, count ] : cellCounts)
    {
        cellOutPoints.push_back(cellOutPoints.back() + count * 3);
    }

    const float* pointFile = (const float*)m_PointFile.data();

    auto min = bb.getMin(), max = bb.getMax();

    #pragma omp parallel for
    for (size_t i = 0; i < cellCounts.size(); i++)
    {
        auto& [ cell, cellNumPoints ] = cellCounts[i];

        float* cellOut = cellOutPoints[i];

        const float* cellIn = pointFile + 3 * cell->offset;
        const float* cellInEnd = cellIn + 3 * cell->size;
        for (const float* p = cellIn; p < cellInEnd; p += 3)
        {
            if (p[0] >= min.x && p[0] <= max.x && p[1] >= min.y && p[1] <= max.y && p[2] >= min.z && p[2] <= max.z)
            {
                cellOut = std::copy_n(p, 3, cellOut);
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
lvr2::floatArr BigGrid<BaseVecT>::normals(const BoundingBox<BaseVecT>& bb, size_t& numNormals, size_t minNumNormals) const
{
    if (!m_hasNormal)
    {
        numNormals = 0;
        return lvr2::floatArr();
    }

    std::vector<std::pair<const CellInfo*, size_t>> cellCounts;
    numNormals = getSizeofBox(bb, cellCounts);

    if (numNormals < minNumNormals)
    {
        return lvr2::floatArr();
    }

    lvr2::floatArr normals(new float[numNormals * 3]);

    // determine where each cell is going to start in the point array
    std::vector<float*> cellOutNormals;
    cellOutNormals.push_back(normals.get());
    for (auto& [ cell, count ] : cellCounts)
    {
        cellOutNormals.push_back(cellOutNormals.back() + count * 3);
    }

    const float* pointFile = (const float*)m_PointFile.data();
    const float* normalFile = (const float*)m_NormalFile.data();

    auto min = bb.getMin(), max = bb.getMax();

    #pragma omp parallel for
    for (size_t i = 0; i < cellCounts.size(); i++)
    {
        auto& [ cell, cellNumNormals ] = cellCounts[i];

        float* cellOut = cellOutNormals[i];

        const float* cellIn = normalFile + 3 * cell->offset;
        const float* points = pointFile + 3 * cell->offset;
        const float* pointsEnd = points + 3 * cell->size;
        for (const float* p = points; p < pointsEnd; p += 3, cellIn += 3)
        {
            if (p[0] >= min.x && p[0] <= max.x && p[1] >= min.y && p[1] <= max.y && p[2] >= min.z && p[2] <= max.z)
            {
                cellOut = std::copy_n(cellIn, 3, cellOut);
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
lvr2::ucharArr BigGrid<BaseVecT>::colors(const BoundingBox<BaseVecT>& bb, size_t& numColors, size_t minNumColors) const
{
    if (!m_hasColor)
    {
        numColors = 0;
        return lvr2::ucharArr();
    }

    std::vector<std::pair<const CellInfo*, size_t>> cellCounts;
    numColors = getSizeofBox(bb, cellCounts);

    if (numColors < minNumColors)
    {
        return lvr2::ucharArr();
    }

    lvr2::ucharArr colors(new float[numColors * 3]);

    // determine where each cell is going to start in the point array
    std::vector<uchar*> cellOutColors;
    cellOutColors.push_back(colors.get());
    for (auto& [ cell, count ] : cellCounts)
    {
        cellOutColors.push_back(cellOutColors.back() + count * 3);
    }

    const float* pointFile = (const float*)m_PointFile.data();
    const uchar* colorFile = (const uchar*)m_ColorFile.data();

    auto min = bb.getMin(), max = bb.getMax();

    #pragma omp parallel for
    for (size_t i = 0; i < cellCounts.size(); i++)
    {
        auto& [ cell, cellNumColors ] = cellCounts[i];

        uchar* cellOut = cellOutColors[i];

        const uchar* cellIn = colorFile + 3 * cell->offset;
        const float* points = pointFile + 3 * cell->offset;
        const float* pointsEnd = points + 3 * cell->size;
        for (const float* p = points; p < pointsEnd; p += 3, cellIn += 3)
        {
            if (p[0] >= min.x && p[0] <= max.x && p[1] >= min.y && p[1] <= max.y && p[2] >= min.z && p[2] <= max.z)
            {
                cellOut = std::copy_n(cellIn, 3, cellOut);
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
lvr2::floatArr BigGrid<BaseVecT>::getPointCloud(size_t& numPoints)
{
    numPoints = m_numPoints;

    lvr2::floatArr points(new float[3 * numPoints]);

    const float* pointData = (const float*)m_PointFile.data();
    std::copy_n(pointData, 3 * numPoints, points.get());

    return points;
}

template <typename BaseVecT>
size_t BigGrid<BaseVecT>::getSizeofBox(const BoundingBox<BaseVecT>& bb, std::vector<std::pair<const CellInfo*, size_t>>& cellCounts) const
{
    auto min = bb.getMin(), max = bb.getMax();
    Vector3i indexMin = calcIndex(min);
    Vector3i indexMax = calcIndex(max);

    cellCounts.clear();

    #pragma omp parallel for
    for (size_t i = 0; i < m_cells.bucket_count(); i++)
    {
        auto start = m_cells.begin(i), end = m_cells.end(i);
        for (auto it = start; it != end; ++it)
        {
            auto& [ index, cell ] = *it;
            if (cell.size == 0)
            {
                continue; // skip extruded cells
            }
            if (index.x() >= indexMin.x() && index.y() >= indexMin.y() && index.z() >= indexMin.z() &&
                index.x() <= indexMax.x() && index.y() <= indexMax.y() && index.z() <= indexMax.z())
            {
                // cells along the boundary need to check individual points
                bool onBoundary = false;
                for (int axis = 0; axis < 3; axis++)
                {
                    if (index[axis] == indexMin[axis] || index[axis] == indexMax[axis])
                    {
                        onBoundary = true;
                        break;
                    }
                }

                #pragma omp critical
                cellCounts.emplace_back(&cell, onBoundary ? 0 : cell.size);
            }
        }
    }

    size_t numPoints = 0;
    const float* pointFile = (const float*)m_PointFile.data();

    #pragma omp parallel for reduction(+:numPoints)
    for (size_t i = 0; i < cellCounts.size(); i++)
    {
        auto& [ cell, cellNumPoints ] = cellCounts[i];
        if (cellNumPoints > 0)
        {
            // cell wasn't a boundary
            numPoints += cellNumPoints;
            continue;
        }

        const float* cellPoints = pointFile + 3 * cell->offset;
        const float* cellPointsEnd = cellPoints + 3 * cell->size;
        for (const float* p = cellPoints; p < cellPointsEnd; p += 3)
        {
            if (p[0] >= min.x && p[0] <= max.x && p[1] >= min.y && p[1] <= max.y && p[2] >= min.z && p[2] <= max.z)
            {
                cellNumPoints++;
            }
        }
        numPoints += cellNumPoints;
    }

    return numPoints;
}

template <typename BaseVecT>
size_t BigGrid<BaseVecT>::estimateSizeofBox(const BoundingBox<BaseVecT>& bb) const
{
    Vector3i indexMin = calcIndex(bb.getMin());
    Vector3i indexMax = calcIndex(bb.getMax());

    size_t numPoints = 0;

    #pragma omp parallel for reduction(+:numPoints)
    for (size_t i = 0; i < m_cells.bucket_count(); i++)
    {
        auto start = m_cells.begin(i), end = m_cells.end(i);
        for (auto it = start; it != end; ++it)
        {
            auto& [ index, cell ] = *it;
            if (index.x() >= indexMin.x() && index.y() >= indexMin.y() && index.z() >= indexMin.z() &&
                index.x() <= indexMax.x() && index.y() <= indexMax.y() && index.z() <= indexMax.z())
            {
                numPoints += cell.size;
            }
        }
    }

    return numPoints;
}

} // namespace lvr2
