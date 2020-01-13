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
 * BigVolumen.tcc
 *
 *  Created on: Jul 17, 2017
 *      Author: Isaak Mitschke
 */

#include "lvr2/io/Timestamp.hpp"
#include "lvr2/io/Progress.hpp"
#include <cstring>
#include "lvr2/io/LineReader.hpp"
#include "lvr2/reconstruction/FastReconstructionTables.hpp"

namespace lvr2
{

template <typename BaseVecT>
BigVolumen<BaseVecT>::BigVolumen(std::vector<std::string> cloudPath, float voxelsize, float overlapping_size, float scale) :
        m_maxIndex(0), m_maxIndexSquare(0), m_maxIndexX(0), m_maxIndexY(0), m_maxIndexZ(0), m_numPoints(0), m_extrude(true),m_scale(scale),
        m_has_normal(false), m_has_color(false)
{
//    exit(-1);
    if(overlapping_size==0) overlapping_size = voxelsize/10;
    m_voxelSize = voxelsize;
    float overlapp_size = overlapping_size;
    //First, parse whole file to get BoundingBox and amount of points
    float ix,iy,iz;
    std::cout << lvr2::timestamp << " Starting BV BB" << std::endl;


    m_numPoints = 0;
    size_t rsize = 0;
    LineReader lineReader(cloudPath);
    cout << "INPUT FILE TYPE: ";
    if(lineReader.getFileType()==XYZ) cout << "XYZ" << endl;
    if(lineReader.getFileType()==XYZN) cout << "XYZN" << endl;
    if(lineReader.getFileType()==XYZRGB) cout << "XYZRGB" << endl;
    if(lineReader.getFileType()==XYZNRGB) cout << "XYZNRGB" << endl;
    size_t lasti = 0;

    while(lineReader.ok())
    {
        if(lineReader.getFileType() == XYZNRGB)
        {
            boost::shared_ptr<xyznc> a = boost::static_pointer_cast<xyznc> (lineReader.getNextPoints(rsize,10000000));
            if (rsize <= 0  && !lineReader.ok())
            {
                break;
            }
            for(int i = 0 ; i< rsize ; i++)
            {
                BaseVecT tmp(a.get()[i].point.x*m_scale, a.get()[i].point.y*m_scale, a.get()[i].point.z*m_scale);
                m_bb.expand(tmp);
                m_numPoints++;
            }
        }
        else if(lineReader.getFileType() == XYZN)
        {
            boost::shared_ptr<xyzn> a = boost::static_pointer_cast<xyzn> (lineReader.getNextPoints(rsize,10000000));
            if (rsize <= 0  && !lineReader.ok())
            {
                break;
            }
            for(int i = 0 ; i< rsize ; i++)
            {
                BaseVecT tmp(a.get()[i].point.x*m_scale, a.get()[i].point.y*m_scale, a.get()[i].point.z*m_scale);
                m_bb.expand(tmp);
                m_numPoints++;
            }
        }
        else if(lineReader.getFileType() == XYZ)
        {
            boost::shared_ptr<xyz> a = boost::static_pointer_cast<xyz> (lineReader.getNextPoints(rsize,10000000));
            if (rsize <= 0  && !lineReader.ok())
            {
                break;
            }
            for(size_t i = 0 ; i< rsize ; i++)
            {
                BaseVecT tmp(a.get()[i].point.x*m_scale, a.get()[i].point.y*m_scale, a.get()[i].point.z*m_scale);
                m_bb.expand(tmp);
                m_numPoints++;
                lasti = i;
            }
        }
        else if(lineReader.getFileType() == XYZRGB)
        {
            boost::shared_ptr<xyzc> a = boost::static_pointer_cast<xyzc> (lineReader.getNextPoints(rsize,10000000));
            if (rsize <= 0  && !lineReader.ok())
            {
                break;
            }
            for(size_t i = 0 ; i< rsize ; i++)
            {
                BaseVecT tmp(a.get()[i].point.x*m_scale, a.get()[i].point.y*m_scale, a.get()[i].point.z*m_scale);
                m_bb.expand(tmp);
                m_numPoints++;
                lasti = i;
            }
        }
        else
        {
            exit(-1);
        }
    }

    cout << lvr2::timestamp << "finished BoundingBox" << endl;

    //Make box side lenghts be divisible by voxel size
    float longestSide = m_bb.getLongestSide();

    BaseVecT center = m_bb.getCentroid();
    size_t xsize2 = calcIndex(m_bb.getXSize ()/m_voxelSize);
    float xsize = ceil(m_bb.getXSize () / voxelsize)*voxelsize;
    float ysize = ceil(m_bb.getYSize () / voxelsize)*voxelsize;
    float zsize = ceil(m_bb.getZSize () / voxelsize)*voxelsize;
    m_bb.expand(BaseVecT(center.x + xsize/2, center.y + ysize/2, center.z + zsize/2));
    m_bb.expand(BaseVecT(center.x - xsize/2, center.y - ysize/2, center.z - zsize/2));
    longestSide = ceil(longestSide / voxelsize) * voxelsize;

    //calc max indices

    //m_maxIndex = (size_t)(longestSide/voxelsize);
    m_maxIndexX = (size_t)(xsize/voxelsize);
    m_maxIndexY = (size_t)(ysize/voxelsize);
    m_maxIndexZ = (size_t)(zsize/voxelsize);
    m_maxIndexX+=1;
    m_maxIndexY+=2;
    m_maxIndexZ+=3;
    m_maxIndex = std::max(m_maxIndexX, std::max(m_maxIndexY,m_maxIndexZ)) ;

    m_maxIndexSquare = m_maxIndex * m_maxIndex;
    std::cout << "BV: \t size: " << m_numPoints << std::endl;
    std::cout << "BB: " << endl << m_bb << endl;
    std::cout << "BV: " << m_maxIndexSquare << "|" << m_maxIndexX << "|" << m_maxIndexY << "|" << m_maxIndexZ << std::endl;


    //
    lineReader.rewind();

    // Bounding Box calculated!!!


    size_t idx, idy, idz;

    if(lineReader.getFileType() == XYZNRGB || lineReader.getFileType() == XYZN)
    {
        m_has_normal = true;
    }
    if(lineReader.getFileType() == XYZNRGB || lineReader.getFileType() == XYZRGB)
    {
        m_has_color = true;
    }

    string comment = lvr2::timestamp.getElapsedTime() + "Building grid... ";
    lvr2::ProgressBar progress(this->m_numPoints, comment);

    std::cout << lvr2::timestamp << "Starting grid generation..." << std::endl;

    while (lineReader.ok())
    {
        if (lineReader.getFileType() == XYZNRGB)
        {
            boost::shared_ptr<xyznc> a = boost::static_pointer_cast<xyznc>(lineReader.getNextPoints(rsize,10000000));
            if (rsize <= 0  && !lineReader.ok())
            {
                break;
            }
            for (int i = 0; i < rsize; i++, ++progress) {
                ix = a.get()[i].point.x * m_scale;
                iy = a.get()[i].point.y * m_scale;
                iz = a.get()[i].point.z * m_scale;

                if(std::isnan(ix) || std::isnan(iy) || std::isnan(iz) )
                {
                    continue;
                }

                size_t idx = calcIndex(fabs(ix - m_bb.getMin()[0]) / voxelsize);
                size_t idy = calcIndex(fabs(iy - m_bb.getMin()[1]) / voxelsize);
                size_t idz = calcIndex(fabs(iz - m_bb.getMin()[2]) / voxelsize);
                size_t h = hashValue(idx, idy, idz);
                m_gridNumPoints[h].ix = idx;
                m_gridNumPoints[h].iy = idy;
                m_gridNumPoints[h].iz = idz;
                m_gridNumPoints[h].size++;
                if (!m_gridNumPoints[h].ofs_points.is_open()) {
                    std::stringstream ss;
                    ss << "part-" << idx << "-" << idy << "-" << idz << "-points.binary";
                    m_gridNumPoints[h].ofs_points.open(ss.str(), std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);
                }

                    // m_gridNumPoints[h].ofs_points << ix << " " << iy << " " << iz << endl;
                m_gridNumPoints[h].ofs_points.write((char*)&ix,sizeof(float));
                m_gridNumPoints[h].ofs_points.write((char*)&iy,sizeof(float));
                m_gridNumPoints[h].ofs_points.write((char*)&iz,sizeof(float));

                if (!m_gridNumPoints[h].ofs_normals.is_open()) {
                    std::stringstream ss;
                    ss << "part-" << idx << "-" << idy << "-" << idz << "-normals.binary";
                    m_gridNumPoints[h].ofs_normals.open(ss.str(), std::ofstream::out |std::ofstream::binary | std::ofstream::trunc);
                }
                // m_gridNumPoints[h].ofs_normals << a.get()[i].normal.x << " " << a.get()[i].normal.y << " "
                //                                << a.get()[i].normal.z << endl;
                m_gridNumPoints[h].ofs_normals.write((char*)&a.get()[i].normal.x,sizeof(float));
                m_gridNumPoints[h].ofs_normals.write((char*)&a.get()[i].normal.y,sizeof(float));
                m_gridNumPoints[h].ofs_normals.write((char*)&a.get()[i].normal.z,sizeof(float));

                if (!m_gridNumPoints[h].ofs_colors.is_open()) {
                    std::stringstream ss;
                    ss << "part-" << idx << "-" << idy << "-" << idz << "-colors.binary";
                    m_gridNumPoints[h].ofs_colors.open(ss.str(), std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);
                }
                // m_gridNumPoints[h].ofs_colors << a.get()[i].color.r << " " << a.get()[i].color.g << " "
                //                               << a.get()[i].color.b << endl;
                m_gridNumPoints[h].ofs_colors.write((char*)&a.get()[i].color.r,sizeof(unsigned char));
                m_gridNumPoints[h].ofs_colors.write((char*)&a.get()[i].color.g,sizeof(unsigned char));
                m_gridNumPoints[h].ofs_colors.write((char*)&a.get()[i].color.b,sizeof(unsigned char));

            }
            for(auto it = m_gridNumPoints.begin(); it!= m_gridNumPoints.end(); ++it)
            {
                if(it->second.ofs_points.is_open())
                {
                    it->second.ofs_points.close();
                }
            }
        } else if (lineReader.getFileType() == XYZN) {
            boost::shared_ptr<xyzn> a = boost::static_pointer_cast<xyzn>(lineReader.getNextPoints(rsize,10000000));
            if (rsize <= 0  && !lineReader.ok())
            {
                break;
            }
            for (int i = 0; i < rsize; i++, ++progress) {
                ix = a.get()[i].point.x * m_scale;
                iy = a.get()[i].point.y * m_scale;
                iz = a.get()[i].point.z * m_scale;

                if(std::isnan(ix) || std::isnan(iy) || std::isnan(iz) )
                {
                    continue;
                }

                size_t idx = calcIndex((ix - m_bb.getMin()[0]) / voxelsize);
                size_t idy = calcIndex((iy - m_bb.getMin()[1]) / voxelsize);
                size_t idz = calcIndex((iz - m_bb.getMin()[2]) / voxelsize);
                size_t h = hashValue(idx, idy, idz);
                m_gridNumPoints[h].ix = idx;
                m_gridNumPoints[h].iy = idy;
                m_gridNumPoints[h].iz = idz;
                m_gridNumPoints[h].size++;
                if (!m_gridNumPoints[h].ofs_points.is_open()) {
                    std::stringstream ss;
                    ss << "part-" << idx << "-" << idy << "-" << idz << "-points.binary";
                    m_gridNumPoints[h].ofs_points.open(ss.str(), std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);
                }

                // m_gridNumPoints[h].ofs_points << ix << " " << iy << " " << iz << endl;
                m_gridNumPoints[h].ofs_points.write((char*)&ix,sizeof(float));
                m_gridNumPoints[h].ofs_points.write((char*)&iy,sizeof(float));
                m_gridNumPoints[h].ofs_points.write((char*)&iz,sizeof(float));

                if (!m_gridNumPoints[h].ofs_normals.is_open()) {
                    std::stringstream ss;
                    ss << "part-" << idx << "-" << idy << "-" << idz << "-normals.binary";
                    m_gridNumPoints[h].ofs_normals.open(ss.str(), std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);
                }


                m_gridNumPoints[h].ofs_normals.write((char*)&a.get()[i].normal.x,sizeof(float));
                m_gridNumPoints[h].ofs_normals.write((char*)&a.get()[i].normal.y,sizeof(float));
                m_gridNumPoints[h].ofs_normals.write((char*)&a.get()[i].normal.z,sizeof(float));

            }
            for(auto it = m_gridNumPoints.begin(); it!= m_gridNumPoints.end(); ++it)
            {
                if(it->second.ofs_points.is_open())
                {
                    it->second.ofs_points.close();
                }
            }
        } else if (lineReader.getFileType() == XYZ)
        {
            boost::shared_ptr<xyz> a = boost::static_pointer_cast<xyz>(lineReader.getNextPoints(rsize,10000000));
            if(rsize>0) std::cout << "RSIZE: " << rsize << endl;
            else std::cout << "rsize :0 " << " lr: " << lineReader.ok() << endl;
//            std::cout << "RSIZE: " << rsize << endl;
            if (rsize <= 0  && !lineReader.ok())
            {
                break;
            }
            for (int i = 0; i < rsize; i++, ++progress) {
                ix = a.get()[i].point.x * m_scale;
                iy = a.get()[i].point.y * m_scale;
                iz = a.get()[i].point.z * m_scale;

                if(std::isnan(ix) || std::isnan(iy) || std::isnan(iz) )
                {
                    continue;
                }

                size_t idx = calcIndex((ix - m_bb.getMin()[0]) / voxelsize);
                size_t idy = calcIndex((iy - m_bb.getMin()[1]) / voxelsize);
                size_t idz = calcIndex((iz - m_bb.getMin()[2]) / voxelsize);
                if(idx > m_maxIndexX || idy > m_maxIndexY || idz > m_maxIndexZ)
                {
                    cout << "FUCK YOU " << idx << "|" << idy << "|" << idz << endl;
                }
                size_t h = hashValue(idx, idy, idz);
                if(m_gridNumPoints[h].size > 0 && (m_gridNumPoints[h].ix != idx || m_gridNumPoints[h].iy != idy || m_gridNumPoints[h].iz != idz)   )
                    cout << "FUCK 2000 " << m_gridNumPoints[h].ix << "|" << m_gridNumPoints[h].iy << "|" << m_gridNumPoints[h].iz << " - " << idx << "|" << idy << "|" <<idz << endl;
                m_gridNumPoints[h].ix = idx;
                m_gridNumPoints[h].iy = idy;
                m_gridNumPoints[h].iz = idz;

                m_gridNumPoints[h].size++;
                if (!m_gridNumPoints[h].ofs_points.is_open()) {
                    std::stringstream ss;
                    ss << "part-" << idx << "-" << idy << "-" << idz << "-points.binary";
//                    cout << ss.str()                    cout << ss.str()
                    size_t streampos = 0;
                    if(m_gridNumPoints[h].size > 0)
                    {
                        m_gridNumPoints[h].ofs_points.open(ss.str(), std::ofstream::out | std::ofstream::binary | std::ofstream::app);
                    }
                    else
                    {
                        m_gridNumPoints[h].ofs_points.open(ss.str(), std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);
                    }

                }


                m_gridNumPoints[h].ofs_points.write((char*)&ix,sizeof(float));
                m_gridNumPoints[h].ofs_points.write((char*)&iy,sizeof(float));
                m_gridNumPoints[h].ofs_points.write((char*)&iz,sizeof(float));

            }
            for(auto it = m_gridNumPoints.begin(); it!= m_gridNumPoints.end(); ++it)
            {
                if(it->second.ofs_points.is_open())
                {
                    it->second.ofs_points.close();
                }
            }

        }
        else if (lineReader.getFileType() == XYZRGB)
        {
            boost::shared_ptr<xyzc> a = boost::static_pointer_cast<xyzc>(lineReader.getNextPoints(rsize,10000000));
            if (rsize <= 0  && !lineReader.ok())
            {
                break;
            }

            for (int i = 0; i < rsize; i++, ++progress)
            {
                ix = a.get()[i].point.x*m_scale;
                iy = a.get()[i].point.y*m_scale;
                iz = a.get()[i].point.z*m_scale;

                if(std::isnan(ix) || std::isnan(iy) || std::isnan(iz) )
                {
                    continue;
                }

                size_t idx = calcIndex((ix - m_bb.getMin()[0])/voxelsize);
                size_t idy = calcIndex((iy - m_bb.getMin()[1])/voxelsize);
                size_t idz = calcIndex((iz - m_bb.getMin()[2])/voxelsize);
                size_t h = hashValue(idx,idy,idz);
                m_gridNumPoints[h].ix = idx;
                m_gridNumPoints[h].iy = idy;
                m_gridNumPoints[h].iz = idz;

                if(!m_gridNumPoints[h].ofs_points.is_open())
                {
                    std::stringstream ss;
                    ss << "part-" << idx << "-" << idy << "-" << idz << "-points.binary";
                    m_gridNumPoints[h].ofs_points.open(ss.str(), std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);
                }

                // stringstream tmpss;
                // tmpss << ix << " " << iy << " " << iz;

                // m_gridNumPoints[h].ofs_points << tmpss.str() << endl;

                m_gridNumPoints[h].ofs_points.write((char*)&ix,sizeof(float));
                m_gridNumPoints[h].ofs_points.write((char*)&iy,sizeof(float));
                m_gridNumPoints[h].ofs_points.write((char*)&iz,sizeof(float));

                m_gridNumPoints[h].size++;

                if(!m_gridNumPoints[h].ofs_colors.is_open())
                {
                    std::stringstream ss;
                    ss << "part-" << idx << "-" << idy << "-" << idz << "-colors.binary";
                    m_gridNumPoints[h].ofs_colors.open(ss.str(), std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);
                }

                m_gridNumPoints[h].ofs_colors.write((char*)&a.get()[i].color.r,sizeof(unsigned char));
                m_gridNumPoints[h].ofs_colors.write((char*)&a.get()[i].color.g,sizeof(unsigned char));
                m_gridNumPoints[h].ofs_colors.write((char*)&a.get()[i].color.b,sizeof(unsigned char));

            }
            for(auto it = m_gridNumPoints.begin(); it!= m_gridNumPoints.end(); ++it)
            {
                if(it->second.ofs_points.is_open())
                {
                    it->second.ofs_points.close();
                }
            }
        }

        progress += rsize;
    }

    // wrote everything to files

    cout << lvr2::timestamp << " calculating boundingboxes of cells" << endl;
    for(auto it = m_gridNumPoints.begin(); it != m_gridNumPoints.end(); it++)
    {
        float cx  = m_bb.getMin().x + it->second.ix * m_voxelSize;
        float cy = m_bb.getMin().y + it->second.iy * m_voxelSize;
        float cz = m_bb.getMin().z + it->second.iz * m_voxelSize;

        float minx = cx-m_voxelSize/2;
        float miny = cy-m_voxelSize/2;
        float minz = cz-m_voxelSize/2;

        float maxx = cx+m_voxelSize/2;
        float maxy = cy+m_voxelSize/2;
        float maxz = cz+m_voxelSize/2;

        BaseVecT min(minx,miny,minz);
        BaseVecT max(maxx,maxy,maxz);

        it->second.bb = BoundingBox<BaseVecT>(min,max);
//        it->second.bb.expand(maxx, maxy, maxz);
//        it->second.bb.expand(minx, miny, minz);
    }
    lineReader.rewind();

    string comment2 = lvr2::timestamp.getElapsedTime() + "adding overlapping points";
    lvr2::ProgressBar progress2(m_numPoints, comment2);
    // Add overlapping points
    for(auto cell = m_gridNumPoints.begin() ; cell != m_gridNumPoints.end(); cell++)
    {
        if (lineReader.getFileType() == XYZ)
        {
            stringstream ss_points;
            ss_points << "part-" << cell->second.ix << "-" << cell->second.iy << "-" << cell->second.iz << "-points.binary";
            ifstream ifs_points(ss_points.str(), std::ifstream::binary);
            lvr2::floatArr pointBuffer(new float[cell->second.size*3]);
            ifs_points.read((char*)pointBuffer.get(), sizeof(float)*3*cell->second.size);
            for(size_t i = 0 ; i<cell->second.size; i++)
            {
                float x = pointBuffer[i*3];
                float y = pointBuffer[i*3+1];
                float z = pointBuffer[i*3+2];
                if(x < cell->second.bb.getMin().x + overlapp_size)
                {
                    size_t neighbour_hash = hashValue(cell->second.ix-1, cell->second.iy, cell->second.iz);
                    auto neigbout_it = m_gridNumPoints.find(neighbour_hash);
                    if(neigbout_it != m_gridNumPoints.end())
                    {
                        neigbout_it->second.overlapping_size++;
                        //neigbout_it->second.ofs_points << x << " " << y << " " << z << endl;
                        neigbout_it->second.ofs_points.write((char*)&pointBuffer[i*3], sizeof(float) * 3 );
                    }
                }
                if(x > cell->second.bb.getMax().x - overlapp_size)
                {
                    size_t neighbour_hash = hashValue(cell->second.ix+1, cell->second.iy, cell->second.iz);
                    auto neigbout_it = m_gridNumPoints.find(neighbour_hash);
                    if(neigbout_it != m_gridNumPoints.end())
                    {
                        neigbout_it->second.overlapping_size++;
                        // neigbout_it->second.ofs_points << x << " " << y << " " << z << endl;
                        neigbout_it->second.ofs_points.write((char*)&pointBuffer[i*3], sizeof(float) * 3 );
                    }
                }
                if(y < cell->second.bb.getMin().y + overlapp_size)
                {
                    size_t neighbour_hash = hashValue(cell->second.ix, cell->second.iy-1, cell->second.iz);
                    auto neigbout_it = m_gridNumPoints.find(neighbour_hash);
                    if(neigbout_it != m_gridNumPoints.end())
                    {
                        neigbout_it->second.overlapping_size++;
                        // neigbout_it->second.ofs_points << x << " " << y << " " << z << endl;
                        neigbout_it->second.ofs_points.write((char*)&pointBuffer[i*3], sizeof(float) * 3 );
                    }
                }
                if(y > cell->second.bb.getMax().y - overlapp_size)
                {
                    size_t neighbour_hash = hashValue(cell->second.ix, cell->second.iy+1, cell->second.iz);
                    auto neigbout_it = m_gridNumPoints.find(neighbour_hash);
                    if(neigbout_it != m_gridNumPoints.end())
                    {
                        neigbout_it->second.overlapping_size++;
                        // neigbout_it->second.ofs_points << x << " " << y << " " << z << endl;
                        neigbout_it->second.ofs_points.write((char*)&pointBuffer[i*3], sizeof(float) * 3 );
                    }
                }
                if(z < cell->second.bb.getMin().z + overlapp_size)
                {
                    size_t neighbour_hash = hashValue(cell->second.ix, cell->second.iy, cell->second.iz-1);
                    auto neigbout_it = m_gridNumPoints.find(neighbour_hash);
                    if(neigbout_it != m_gridNumPoints.end())
                    {
                        neigbout_it->second.overlapping_size++;
                        // neigbout_it->second.ofs_points << x << " " << y << " " << z << endl;
                        neigbout_it->second.ofs_points.write((char*)&pointBuffer[i*3], sizeof(float) * 3 );
                    }
                }
                if(z > cell->second.bb.getMax().z - overlapp_size)
                {
                    size_t neighbour_hash = hashValue(cell->second.ix, cell->second.iy, cell->second.iz+1);
                    auto neigbout_it = m_gridNumPoints.find(neighbour_hash);
                    if(neigbout_it != m_gridNumPoints.end())
                    {
                        neigbout_it->second.overlapping_size++;
                        // neigbout_it->second.ofs_points << x << " " << y << " " << z << endl;
                        neigbout_it->second.ofs_points.write((char*)&pointBuffer[i*3], sizeof(float) * 3 );
                    }
                }
                ++progress2;
            }
        }
        else if (lineReader.getFileType() == XYZNRGB)
        {
            stringstream ss_points;
            ss_points << "part-" << cell->second.ix << "-" << cell->second.iy << "-" << cell->second.iz << "-points.binary";
            ifstream ifs_points(ss_points.str(), std::ifstream::binary);
            lvr2::floatArr pointBuffer(new float[cell->second.size*3]);
            ifs_points.read((char*)pointBuffer.get(), sizeof(float)*3*cell->second.size);

            stringstream ss_normals;
            ss_normals << "part-" << cell->second.ix << "-" << cell->second.iy << "-" << cell->second.iz << "-normals.binary";
            ifstream ifs_normals(ss_normals.str(), std::ifstream::binary);
            lvr2::floatArr normalBuffer(new float[cell->second.size*3]);
            ifs_normals.read((char*)normalBuffer.get(), sizeof(float)*3*cell->second.size);

            stringstream ss_colors;
            ss_colors << "part-" << cell->second.ix << "-" << cell->second.iy << "-" << cell->second.iz << "-colors.binary";
            ifstream ifs_colors(ss_colors.str(), std::ifstream::binary);
            lvr2::ucharArr colorBuffer(new unsigned char[cell->second.size*3]);
            ifs_colors.read((char*)colorBuffer.get(), sizeof(unsigned char)*3*cell->second.size);

            for(size_t i = 0 ; i<cell->second.size; i++)
            {
                float x = pointBuffer[i*3];
                float y = pointBuffer[i*3+1];
                float z = pointBuffer[i*3+2];
                if(x < cell->second.bb.getMin().x + overlapp_size)
                {
                    size_t neighbour_hash = hashValue(cell->second.ix-1, cell->second.iy, cell->second.iz);
                    auto neigbout_it = m_gridNumPoints.find(neighbour_hash);
                    if(neigbout_it != m_gridNumPoints.end())
                    {
                        neigbout_it->second.overlapping_size++;
                        // neigbout_it->second.ofs_points << x << " " << y << " " << z << endl;
                        neigbout_it->second.ofs_points.write((char*)&pointBuffer[i*3],sizeof(float)*3);

                        // neigbout_it->second.ofs_points << normalBuffer[i*3] << " " << normalBuffer[i*3+1] << " " << normalBuffer[i*3+2]<< endl;
                        neigbout_it->second.ofs_normals.write((char*)&normalBuffer[i*3],sizeof(float) * 3);

                        // neigbout_it->second.ofs_points << colorBuffer[i*3] << " " << colorBuffer[i*3+1] << " " << colorBuffer[i*3+2]<< endl;
                        neigbout_it->second.ofs_colors.write((char*)&colorBuffer[i*3],sizeof(unsigned char) * 3);
                    }
                }
                if(x > cell->second.bb.getMax().x - overlapp_size)
                {
                    size_t neighbour_hash = hashValue(cell->second.ix+1, cell->second.iy, cell->second.iz);
                    auto neigbout_it = m_gridNumPoints.find(neighbour_hash);
                    if(neigbout_it != m_gridNumPoints.end())
                    {
                        neigbout_it->second.overlapping_size++;
                        // neigbout_it->second.ofs_points << x << " " << y << " " << z << endl;
                        neigbout_it->second.ofs_points.write((char*)&pointBuffer[i*3],sizeof(float)*3);

                        // neigbout_it->second.ofs_points << normalBuffer[i*3] << " " << normalBuffer[i*3+1] << " " << normalBuffer[i*3+2]<< endl;
                        neigbout_it->second.ofs_normals.write((char*)&normalBuffer[i*3],sizeof(float) * 3);

                        // neigbout_it->second.ofs_points << colorBuffer[i*3] << " " << colorBuffer[i*3+1] << " " << colorBuffer[i*3+2]<< endl;
                        neigbout_it->second.ofs_colors.write((char*)&colorBuffer[i*3],sizeof(unsigned char) * 3);
                    }
                }
                if(y < cell->second.bb.getMin().y + overlapp_size)
                {
                    size_t neighbour_hash = hashValue(cell->second.ix, cell->second.iy-1, cell->second.iz);
                    auto neigbout_it = m_gridNumPoints.find(neighbour_hash);
                    if(neigbout_it != m_gridNumPoints.end())
                    {
                        neigbout_it->second.overlapping_size++;
                        // neigbout_it->second.ofs_points << x << " " << y << " " << z << endl;
                        neigbout_it->second.ofs_points.write((char*)&pointBuffer[i*3],sizeof(float)*3);

                        // neigbout_it->second.ofs_points << normalBuffer[i*3] << " " << normalBuffer[i*3+1] << " " << normalBuffer[i*3+2]<< endl;
                        neigbout_it->second.ofs_normals.write((char*)&normalBuffer[i*3],sizeof(float) * 3);

                        // neigbout_it->second.ofs_points << colorBuffer[i*3] << " " << colorBuffer[i*3+1] << " " << colorBuffer[i*3+2]<< endl;
                        neigbout_it->second.ofs_colors.write((char*)&colorBuffer[i*3],sizeof(unsigned char) * 3);
                    }
                }
                if(y > cell->second.bb.getMax().y - overlapp_size)
                {
                    size_t neighbour_hash = hashValue(cell->second.ix, cell->second.iy+1, cell->second.iz);
                    auto neigbout_it = m_gridNumPoints.find(neighbour_hash);
                    if(neigbout_it != m_gridNumPoints.end())
                    {
                        neigbout_it->second.overlapping_size++;
                        // neigbout_it->second.ofs_points << x << " " << y << " " << z << endl;
                        neigbout_it->second.ofs_points.write((char*)&pointBuffer[i*3],sizeof(float)*3);

                        // neigbout_it->second.ofs_points << normalBuffer[i*3] << " " << normalBuffer[i*3+1] << " " << normalBuffer[i*3+2]<< endl;
                        neigbout_it->second.ofs_normals.write((char*)&normalBuffer[i*3],sizeof(float) * 3);

                        // neigbout_it->second.ofs_points << colorBuffer[i*3] << " " << colorBuffer[i*3+1] << " " << colorBuffer[i*3+2]<< endl;
                        neigbout_it->second.ofs_colors.write((char*)&colorBuffer[i*3],sizeof(unsigned char) * 3);
                    }
                }
                if(z < cell->second.bb.getMin().z + overlapp_size)
                {
                    size_t neighbour_hash = hashValue(cell->second.ix, cell->second.iy, cell->second.iz-1);
                    auto neigbout_it = m_gridNumPoints.find(neighbour_hash);
                    if(neigbout_it != m_gridNumPoints.end())
                    {
                        neigbout_it->second.overlapping_size++;
                        // neigbout_it->second.ofs_points << x << " " << y << " " << z << endl;
                        neigbout_it->second.ofs_points.write((char*)&pointBuffer[i*3],sizeof(float)*3);

                        // neigbout_it->second.ofs_points << normalBuffer[i*3] << " " << normalBuffer[i*3+1] << " " << normalBuffer[i*3+2]<< endl;
                        neigbout_it->second.ofs_normals.write((char*)&normalBuffer[i*3],sizeof(float) * 3);

                        // neigbout_it->second.ofs_points << colorBuffer[i*3] << " " << colorBuffer[i*3+1] << " " << colorBuffer[i*3+2]<< endl;
                        neigbout_it->second.ofs_colors.write((char*)&colorBuffer[i*3],sizeof(unsigned char) * 3);
                    }
                }
                if(z > cell->second.bb.getMin().z - overlapp_size)
                {
                    size_t neighbour_hash = hashValue(cell->second.ix, cell->second.iy, cell->second.iz+1);
                    auto neigbout_it = m_gridNumPoints.find(neighbour_hash);
                    if(neigbout_it != m_gridNumPoints.end())
                    {
                        neigbout_it->second.overlapping_size++;
                        // neigbout_it->second.ofs_points << x << " " << y << " " << z << endl;
                        neigbout_it->second.ofs_points.write((char*)&pointBuffer[i*3],sizeof(float)*3);

                        // neigbout_it->second.ofs_points << normalBuffer[i*3] << " " << normalBuffer[i*3+1] << " " << normalBuffer[i*3+2]<< endl;
                        neigbout_it->second.ofs_normals.write((char*)&normalBuffer[i*3],sizeof(float) * 3);

                        // neigbout_it->second.ofs_points << colorBuffer[i*3] << " " << colorBuffer[i*3+1] << " " << colorBuffer[i*3+2]<< endl;
                        neigbout_it->second.ofs_colors.write((char*)&colorBuffer[i*3],sizeof(unsigned char) * 3);
                    }
                }
                ++progress2;
            }
        }
        else if (lineReader.getFileType() == XYZN)
        {
            stringstream ss_points;
            ss_points << "part-" << cell->second.ix << "-" << cell->second.iy << "-" << cell->second.iz << "-points.binary";
            ifstream ifs_points(ss_points.str(), std::ifstream::binary);
            lvr2::floatArr pointBuffer(new float[cell->second.size*3]);
            ifs_points.read((char*)pointBuffer.get(), sizeof(float)*3*cell->second.size);

            stringstream ss_normals;
            ss_normals << "part-" << cell->second.ix << "-" << cell->second.iy << "-" << cell->second.iz << "-normals.binary";
            ifstream ifs_normals(ss_normals.str(), std::ifstream::binary);
            lvr2::floatArr normalBuffer(new float[cell->second.size*3]);
            ifs_normals.read((char*)normalBuffer.get(), sizeof(float)*3*cell->second.size);



            for(size_t i = 0 ; i<cell->second.size; i++)
            {
                float x = pointBuffer[i*3];
                float y = pointBuffer[i*3+1];
                float z = pointBuffer[i*3+2];

                if(std::isnan(x) || std::isnan(y) || std::isnan(z) )
                {
                    continue;
                }
                if(x < cell->second.bb.getMin().x + overlapp_size)
                {
                    size_t neighbour_hash = hashValue(cell->second.ix-1, cell->second.iy, cell->second.iz);
                    auto neigbout_it = m_gridNumPoints.find(neighbour_hash);
                    if(neigbout_it != m_gridNumPoints.end())
                    {
                        neigbout_it->second.overlapping_size++;
                        // neigbout_it->second.ofs_points << x << " " << y << " " << z << endl;
                        neigbout_it->second.ofs_points.write((char*)&pointBuffer[i*3],sizeof(float)*3);

                        // neigbout_it->second.ofs_points << normalBuffer[i*3] << " " << normalBuffer[i*3+1] << " " << normalBuffer[i*3+2]<< endl;
                        neigbout_it->second.ofs_normals.write((char*)&normalBuffer[i*3],sizeof(float) * 3);

                    }
                }
                if(x > cell->second.bb.getMax().x - overlapp_size)
                {
                    size_t neighbour_hash = hashValue(cell->second.ix+1, cell->second.iy, cell->second.iz);
                    auto neigbout_it = m_gridNumPoints.find(neighbour_hash);
                    if(neigbout_it != m_gridNumPoints.end())
                    {
                        neigbout_it->second.overlapping_size++;
                        // neigbout_it->second.ofs_points << x << " " << y << " " << z << endl;
                        neigbout_it->second.ofs_points.write((char*)&pointBuffer[i*3],sizeof(float)*3);

                        // neigbout_it->second.ofs_points << normalBuffer[i*3] << " " << normalBuffer[i*3+1] << " " << normalBuffer[i*3+2]<< endl;
                        neigbout_it->second.ofs_normals.write((char*)&normalBuffer[i*3],sizeof(float) * 3);

                    }
                }
                if(y < cell->second.bb.getMin().y + overlapp_size)
                {
                    size_t neighbour_hash = hashValue(cell->second.ix, cell->second.iy-1, cell->second.iz);
                    auto neigbout_it = m_gridNumPoints.find(neighbour_hash);
                    if(neigbout_it != m_gridNumPoints.end())
                    {
                        neigbout_it->second.overlapping_size++;
                        // neigbout_it->second.ofs_points << x << " " << y << " " << z << endl;
                        neigbout_it->second.ofs_points.write((char*)&pointBuffer[i*3],sizeof(float)*3);

                        // neigbout_it->second.ofs_points << normalBuffer[i*3] << " " << normalBuffer[i*3+1] << " " << normalBuffer[i*3+2]<< endl;
                        neigbout_it->second.ofs_normals.write((char*)&normalBuffer[i*3],sizeof(float) * 3);

                    }
                }
                if(y > cell->second.bb.getMax().y - overlapp_size)
                {
                    size_t neighbour_hash = hashValue(cell->second.ix+1, cell->second.iy, cell->second.iz);
                    auto neigbout_it = m_gridNumPoints.find(neighbour_hash);
                    if(neigbout_it != m_gridNumPoints.end())
                    {
                        neigbout_it->second.overlapping_size++;
                        // neigbout_it->second.ofs_points << x << " " << y << " " << z << endl;
                        neigbout_it->second.ofs_points.write((char*)&pointBuffer[i*3],sizeof(float)*3);

                        // neigbout_it->second.ofs_points << normalBuffer[i*3] << " " << normalBuffer[i*3+1] << " " << normalBuffer[i*3+2]<< endl;
                        neigbout_it->second.ofs_normals.write((char*)&normalBuffer[i*3],sizeof(float) * 3);

                    }
                }
                if(z < cell->second.bb.getMin().z + overlapp_size)
                {
                    size_t neighbour_hash = hashValue(cell->second.ix, cell->second.iy, cell->second.iz-1);
                    auto neigbout_it = m_gridNumPoints.find(neighbour_hash);
                    if(neigbout_it != m_gridNumPoints.end())
                    {
                        neigbout_it->second.overlapping_size++;
                        // neigbout_it->second.ofs_points << x << " " << y << " " << z << endl;
                        neigbout_it->second.ofs_points.write((char*)&pointBuffer[i*3],sizeof(float)*3);

                        // neigbout_it->second.ofs_points << normalBuffer[i*3] << " " << normalBuffer[i*3+1] << " " << normalBuffer[i*3+2]<< endl;
                        neigbout_it->second.ofs_normals.write((char*)&normalBuffer[i*3],sizeof(float) * 3);

                    }
                }
                if(z > cell->second.bb.getMax().z - overlapp_size)
                {
                    size_t neighbour_hash = hashValue(cell->second.ix, cell->second.iy, cell->second.iz+1);
                    auto neigbout_it = m_gridNumPoints.find(neighbour_hash);
                    if(neigbout_it != m_gridNumPoints.end())
                    {
                        neigbout_it->second.overlapping_size++;
                        // neigbout_it->second.ofs_points << x << " " << y << " " << z << endl;
                        neigbout_it->second.ofs_points.write((char*)&pointBuffer[i*3],sizeof(float)*3);

                        // neigbout_it->second.ofs_points << normalBuffer[i*3] << " " << normalBuffer[i*3+1] << " " << normalBuffer[i*3+2]<< endl;
                        neigbout_it->second.ofs_normals.write((char*)&normalBuffer[i*3],sizeof(float) * 3);
                    }
                }
                ++progress2;
            }
        }
        else if (lineReader.getFileType() == XYZRGB)
        {
            stringstream ss_points;
            ss_points << "part-" << cell->second.ix << "-" << cell->second.iy << "-" << cell->second.iz << "-points.binary";
            ifstream ifs_points(ss_points.str(), std::ifstream::binary);
            lvr2::floatArr pointBuffer(new float[cell->second.size*3]);
            ifs_points.read((char*)pointBuffer.get(), sizeof(float)*3*cell->second.size);

            stringstream ss_colors;
            ss_colors << "part-" << cell->second.ix << "-" << cell->second.iy << "-" << cell->second.iz << "-colors.binary";
            ifstream ifs_colors(ss_colors.str(), std::ifstream::binary);
            lvr2::ucharArr colorBuffer(new unsigned char[cell->second.size*3]);
            ifs_colors.read((char*)colorBuffer.get(), sizeof(unsigned char)*3*cell->second.size);

            for(size_t i = 0 ; i<cell->second.size; i++)
            {
                float x = pointBuffer[i*3];
                float y = pointBuffer[i*3+1];
                float z = pointBuffer[i*3+2];
                if( std::isnan(x) || std::isnan(y) && std::isnan(z) )
                {
                    continue;
                }

                if(x < cell->second.bb.getMin().x + overlapp_size)
                {
                    size_t neighbour_hash = hashValue(cell->second.ix-1, cell->second.iy, cell->second.iz);
                    auto neigbout_it = m_gridNumPoints.find(neighbour_hash);
                    if(neigbout_it != m_gridNumPoints.end())
                    {
                        neigbout_it->second.overlapping_size++;
                        // neigbout_it->second.ofs_points << x << " " << y << " " << z << endl;
                        neigbout_it->second.ofs_points.write((char*)&pointBuffer[i*3],sizeof(float)*3);

                        // neigbout_it->second.ofs_points << colorBuffer[i*3] << " " << colorBuffer[i*3+1] << " " << colorBuffer[i*3+2]<< endl;
                        neigbout_it->second.ofs_colors.write((char*)&colorBuffer[i*3],sizeof(unsigned char) * 3);
                    }
                }
                if(x > cell->second.bb.getMax().x - overlapp_size)
                {
                    size_t neighbour_hash = hashValue(cell->second.ix+1, cell->second.iy, cell->second.iz);
                    auto neigbout_it = m_gridNumPoints.find(neighbour_hash);
                    if(neigbout_it != m_gridNumPoints.end())
                    {
                       neigbout_it->second.overlapping_size++;
                        // neigbout_it->second.ofs_points << x << " " << y << " " << z << endl;
                        neigbout_it->second.ofs_points.write((char*)&pointBuffer[i*3],sizeof(float)*3);

                        // neigbout_it->second.ofs_points << colorBuffer[i*3] << " " << colorBuffer[i*3+1] << " " << colorBuffer[i*3+2]<< endl;
                        neigbout_it->second.ofs_colors.write((char*)&colorBuffer[i*3],sizeof(unsigned char) * 3);
                    }
                }
                if(y < cell->second.bb.getMin().y + overlapp_size)
                {
                    size_t neighbour_hash = hashValue(cell->second.ix, cell->second.iy-1, cell->second.iz);
                    auto neigbout_it = m_gridNumPoints.find(neighbour_hash);
                    if(neigbout_it != m_gridNumPoints.end())
                    {
                        neigbout_it->second.overlapping_size++;
                        // neigbout_it->second.ofs_points << x << " " << y << " " << z << endl;
                        neigbout_it->second.ofs_points.write((char*)&pointBuffer[i*3],sizeof(float)*3);

                        // neigbout_it->second.ofs_points << colorBuffer[i*3] << " " << colorBuffer[i*3+1] << " " << colorBuffer[i*3+2]<< endl;
                        neigbout_it->second.ofs_colors.write((char*)&colorBuffer[i*3],sizeof(unsigned char) * 3);
                    }
                }
                if(y > cell->second.bb.getMax().y - overlapp_size)
                {
                    size_t neighbour_hash = hashValue(cell->second.ix, cell->second.iy+1, cell->second.iz);
                    auto neigbout_it = m_gridNumPoints.find(neighbour_hash);
                    if(neigbout_it != m_gridNumPoints.end())
                    {
                        neigbout_it->second.overlapping_size++;
                        // neigbout_it->second.ofs_points << x << " " << y << " " << z << endl;
                        neigbout_it->second.ofs_points.write((char*)&pointBuffer[i*3],sizeof(float)*3);

                        // neigbout_it->second.ofs_points << colorBuffer[i*3] << " " << colorBuffer[i*3+1] << " " << colorBuffer[i*3+2]<< endl;
                        neigbout_it->second.ofs_colors.write((char*)&colorBuffer[i*3],sizeof(unsigned char) * 3);
                    }
                }
                if(z < cell->second.bb.getMin().z + overlapp_size)
                {
                    size_t neighbour_hash = hashValue(cell->second.ix, cell->second.iy, cell->second.iz-1);
                    auto neigbout_it = m_gridNumPoints.find(neighbour_hash);
                    if(neigbout_it != m_gridNumPoints.end())
                    {
                        neigbout_it->second.overlapping_size++;
                        // neigbout_it->second.ofs_points << x << " " << y << " " << z << endl;
                        neigbout_it->second.ofs_points.write((char*)&pointBuffer[i*3],sizeof(float)*3);

                        // neigbout_it->second.ofs_points << colorBuffer[i*3] << " " << colorBuffer[i*3+1] << " " << colorBuffer[i*3+2]<< endl;
                        neigbout_it->second.ofs_colors.write((char*)&colorBuffer[i*3],sizeof(unsigned char) * 3);
                    }
                }
                if(z > cell->second.bb.getMax().z - overlapp_size)
                {
                    size_t neighbour_hash = hashValue(cell->second.ix, cell->second.iy, cell->second.iz+1);
                    auto neigbout_it = m_gridNumPoints.find(neighbour_hash);
                    if(neigbout_it != m_gridNumPoints.end())
                    {
                        neigbout_it->second.overlapping_size++;
                        // neigbout_it->second.ofs_points << x << " " << y << " " << z << endl;
                        neigbout_it->second.ofs_points.write((char*)&pointBuffer[i*3],sizeof(float)*3);

                        // neigbout_it->second.ofs_points << colorBuffer[i*3] << " " << colorBuffer[i*3+1] << " " << colorBuffer[i*3+2]<< endl;
                        neigbout_it->second.ofs_colors.write((char*)&colorBuffer[i*3],sizeof(unsigned char) * 3);
                    }
                }
                ++progress2;
            }
        }
    }
    cout << lvr2::timestamp << " finished serialization" << endl;
    for(auto cell = m_gridNumPoints.begin() ; cell != m_gridNumPoints.end(); cell++)
    {
        if(cell->second.ofs_points.is_open()) cell->second.ofs_points.close();
        if(cell->second.ofs_normals.is_open()) cell->second.ofs_normals.close();
        if(cell->second.ofs_colors.is_open()) cell->second.ofs_colors.close();
    }
}

template <typename BaseVecT>
BigVolumen<BaseVecT>::~BigVolumen() { }

template <typename BaseVecT>
size_t BigVolumen<BaseVecT>::size()
{
    return m_gridNumPoints.size();
}

template <typename BaseVecT>
size_t BigVolumen<BaseVecT>::pointSize()
{
    return m_numPoints;
}

} // namespace lvr2