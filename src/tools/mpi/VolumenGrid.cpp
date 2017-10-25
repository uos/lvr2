//
// Created by imitschke on 18.10.17.
//
#include "LineReader.hpp"
#include "VolumenGrid.h"
#include "lvr/io/Timestamp.hpp"
#include <cstring>
#include "LineReader.hpp"
#include <lvr/reconstruction/FastReconstructionTables.hpp>
using namespace lvr;

VolumenGrid::VolumenGrid(std::string cloudPath, float volumen) : m_maxIndex(0), m_maxIndexSquare(0), m_maxIndexX(0), m_maxIndexY(0), m_maxIndexZ(0), m_numPoints(0), m_extrude(false),m_scale(1),
                                                                m_has_normal(false), m_has_color(false), m_volumen(volumen)
{
    //First, parse whole file to get BoundingBox and amount of points
    std::cout << "opening: "  << cloudPath << endl;
    float ix,iy,iz;
    std::cout << lvr::timestamp << " Starting BB" << std::endl;
    m_numPoints = 0;
    size_t rsize = 0;
    LineReader lineReader(cloudPath);
    size_t lasti = 0;
    while(true)
    {
        if(lineReader.getFileType() == XYZNRGB)
        {
            boost::shared_ptr<xyznc> a = boost::static_pointer_cast<xyznc> (lineReader.getNextPoints(rsize,1024));
            if (rsize <= 0 )
            {
                break;
            }
            for(int i = 0 ; i< rsize ; i++)
            {
                m_bb.expand(a.get()[i].point.x*m_scale,a.get()[i].point.y*m_scale,a.get()[i].point.z*m_scale);
                m_numPoints++;
            }
        }
        else if(lineReader.getFileType() == XYZN)
        {
            boost::shared_ptr<xyzn> a = boost::static_pointer_cast<xyzn> (lineReader.getNextPoints(rsize,1024));
            if (rsize <= 0 )
            {
                break;
            }
            for(int i = 0 ; i< rsize ; i++)
            {
                m_bb.expand(a.get()[i].point.x*m_scale,a.get()[i].point.y*m_scale,a.get()[i].point.z*m_scale);
                m_numPoints++;
            }
        }
        else if(lineReader.getFileType() == XYZ)
        {
            boost::shared_ptr<xyz> a = boost::static_pointer_cast<xyz> (lineReader.getNextPoints(rsize,1024));
            if (rsize <= 0 )
            {
                break;
            }
            for(size_t i = 0 ; i< rsize ; i++)
            {
                m_bb.expand(a.get()[i].point.x*m_scale,a.get()[i].point.y*m_scale,a.get()[i].point.z*m_scale);
                m_numPoints++;
                lasti = i;
            }
        }
        else if(lineReader.getFileType() == XYZRGB)
        {
            boost::shared_ptr<xyzc> a = boost::static_pointer_cast<xyzc> (lineReader.getNextPoints(rsize,1024));
            if (rsize <= 0 )
            {
                break;
            }
            for(size_t i = 0 ; i< rsize ; i++)
            {
                m_bb.expand(a.get()[i].point.x*m_scale,a.get()[i].point.y*m_scale,a.get()[i].point.z*m_scale);
                m_numPoints++;
                lasti = i;
            }
        }
        else
        {
            exit(-1);
        }
    }





    //Make box side lenghts be divisible by voxel size
    float longestSide = m_bb.getLongestSide();

    lvr::Vertexf center = m_bb.getCentroid();
    size_t xsize2 = calcIndex(m_bb.getXSize ()/volumen);
    float xsize = ceil(m_bb.getXSize () / volumen)*volumen;
    float ysize = ceil(m_bb.getYSize () / volumen)*volumen;
    float zsize = ceil(m_bb.getZSize () / volumen)*volumen;
    m_bb.expand(center.x + xsize/2, center.y + ysize/2, center.z + zsize/2  );
    m_bb.expand(center.x - xsize/2, center.y - ysize/2, center.z - zsize/2  );
    longestSide = ceil(longestSide / volumen) * volumen;

    //calc max indices

    //m_maxIndex = (size_t)(longestSide/volumen);
    m_maxIndexX = (size_t)(xsize/volumen);
    m_maxIndexY = (size_t)(ysize/volumen);
    m_maxIndexZ = (size_t)(zsize/volumen);
    m_maxIndex = std::max(m_maxIndexX, std::max(m_maxIndexY,m_maxIndexZ)) + 5*volumen ;
    m_maxIndexX+=1;
    m_maxIndexY+=2;
    m_maxIndexZ+=3;
    m_maxIndexSquare = m_maxIndex * m_maxIndex;
    std::cout << "BG: " << m_maxIndexSquare << "|" << m_maxIndexX << "|" << m_maxIndexY << "|" << m_maxIndexZ << std::endl;
    lineReader.rewind();
    size_t idx, idy, idz;
    while (true) {
        if (lineReader.getFileType() == XYZNRGB) {
            boost::shared_ptr<xyznc> a = boost::static_pointer_cast<xyznc>(lineReader.getNextPoints(rsize,1024));
            if (rsize <= 0) {
                break;
            }
            int dx, dy, dz;
            for (int i = 0; i < rsize; i++) {
                ix = a.get()[i].point.x*m_scale;
                iy = a.get()[i].point.y*m_scale;
                iz = a.get()[i].point.z*m_scale;
                idx = calcIndex((ix - m_bb.getMin()[0])/volumen);
                idy = calcIndex((iy - m_bb.getMin()[1])/volumen);
                idz = calcIndex((iz - m_bb.getMin()[2])/volumen);
                int e;
                this->m_extrude ? e = 8 : e = 1;
                for(int j = 0; j < e; j++)
                {
                    dx = HGCreateTable[j][0];
                    dy = HGCreateTable[j][1];
                    dz = HGCreateTable[j][2];
                    size_t h = hashValue(idx+dx,idy+dy,idz+dz);
                    if( j == 0) m_gridNumPoints[h].size++;
                    else
                    {
                        auto it =  m_gridNumPoints.find(h);
                        if(it == m_gridNumPoints.end())
                        {
                            m_gridNumPoints[h].size = 0;
                            Vertexf center(
                                (idx + dx) * this->m_volumen + m_bb.getMin()[0],
                                (idx + dy) * this->m_volumen + m_bb.getMin()[1],
                                (idx + dz) * this->m_volumen + m_bb.getMin()[2]);
                            Vertexf min(center[0]-(m_volumen/2),center[1]-(m_volumen/2),center[2]-(m_volumen/2) );
                            Vertexf max(center[0]+(m_volumen/2),center[1]+(m_volumen/2),center[2]+(m_volumen/2) );
                            m_gridNumPoints[h].bb.expand(min);
                            m_gridNumPoints[h].bb.expand(max);
                        }
                    }

                }
            }
        } else if (lineReader.getFileType() == XYZN) {
            boost::shared_ptr<xyzn> a = boost::static_pointer_cast<xyzn>(lineReader.getNextPoints(rsize,1024));
            if (rsize <= 0) {
                break;
            }
            int dx, dy, dz;
            for (int i = 0; i < rsize; i++) {
                ix = a.get()[i].point.x*m_scale;
                iy = a.get()[i].point.y*m_scale;
                iz = a.get()[i].point.z*m_scale;
                idx = calcIndex((ix - m_bb.getMin()[0])/volumen);
                idy = calcIndex((iy - m_bb.getMin()[1])/volumen);
                idz = calcIndex((iz - m_bb.getMin()[2])/volumen);

                int e;
                this->m_extrude ? e = 8 : e = 1;
                for(int j = 0; j < e; j++)
                {
                    dx = HGCreateTable[j][0];
                    dy = HGCreateTable[j][1];
                    dz = HGCreateTable[j][2];
                    size_t h = hashValue(idx+dx,idy+dy,idz+dz);
                    if( j == 0) m_gridNumPoints[h].size++;
                    else
                    {
                        auto it =  m_gridNumPoints.find(h);
                        if(it == m_gridNumPoints.end())
                        {
                            m_gridNumPoints[h].size = 0;
                            Vertexf center(
                                    (idx + dx) * this->m_volumen + m_bb.getMin()[0],
                                    (idx + dy) * this->m_volumen + m_bb.getMin()[1],
                                    (idx + dz) * this->m_volumen + m_bb.getMin()[2]);
                            Vertexf min(center[0]-(m_volumen/2),center[1]-(m_volumen/2),center[2]-(m_volumen/2) );
                            Vertexf max(center[0]+(m_volumen/2),center[1]+(m_volumen/2),center[2]+(m_volumen/2) );
                            m_gridNumPoints[h].bb.expand(min);
                            m_gridNumPoints[h].bb.expand(max);
                        }
                    }

                }



            }
        } else if (lineReader.getFileType() == XYZ) {
            boost::shared_ptr<xyz> a = boost::static_pointer_cast<xyz>(lineReader.getNextPoints(rsize,1024));
            if (rsize <= 0) {
                break;
            }
            int dx, dy, dz;
            for (int i = 0; i < rsize; i++) {
                ix = a.get()[i].point.x*m_scale;
                iy = a.get()[i].point.y*m_scale;
                iz = a.get()[i].point.z*m_scale;
                idx = calcIndex((ix - m_bb.getMin()[0])/volumen);
                idy = calcIndex((iy - m_bb.getMin()[1])/volumen);
                idz = calcIndex((iz - m_bb.getMin()[2])/volumen);
                int e;
                this->m_extrude ? e = 8 : e = 1;
                for(int j = 0; j < e; j++)
                {
                    dx = HGCreateTable[j][0];
                    dy = HGCreateTable[j][1];
                    dz = HGCreateTable[j][2];
                    size_t h = hashValue(idx+dx,idy+dy,idz+dz);
                    if( j == 0) m_gridNumPoints[h].size++;
                    else
                    {
                        auto it =  m_gridNumPoints.find(h);
                        if(it == m_gridNumPoints.end())
                        {
                            m_gridNumPoints[h].size = 0;
                            Vertexf center(
                                    (idx + dx) * this->m_volumen + m_bb.getMin()[0],
                                    (idx + dy) * this->m_volumen + m_bb.getMin()[1],
                                    (idx + dz) * this->m_volumen + m_bb.getMin()[2]);
                            Vertexf min(center[0]-(m_volumen/2),center[1]-(m_volumen/2),center[2]-(m_volumen/2) );
                            Vertexf max(center[0]+(m_volumen/2),center[1]+(m_volumen/2),center[2]+(m_volumen/2) );
                            m_gridNumPoints[h].bb.expand(min);
                            m_gridNumPoints[h].bb.expand(max);
                        }
                    }

                }
            }
        }
        else if (lineReader.getFileType() == XYZRGB) {
            boost::shared_ptr<xyzc> a = boost::static_pointer_cast<xyzc>(lineReader.getNextPoints(rsize,1024));
            if (rsize <= 0) {
                break;
            }
            int dx, dy, dz;
            for (int i = 0; i < rsize; i++) {
                ix = a.get()[i].point.x*m_scale;
                iy = a.get()[i].point.y*m_scale;
                iz = a.get()[i].point.z*m_scale;
                idx = calcIndex((ix - m_bb.getMin()[0])/volumen);
                idy = calcIndex((iy - m_bb.getMin()[1])/volumen);
                idz = calcIndex((iz - m_bb.getMin()[2])/volumen);
                int e;
                this->m_extrude ? e = 8 : e = 1;
                for(int j = 0; j < e; j++)
                {
                    dx = HGCreateTable[j][0];
                    dy = HGCreateTable[j][1];
                    dz = HGCreateTable[j][2];
                    size_t h = hashValue(idx+dx,idy+dy,idz+dz);
                    if( j == 0) m_gridNumPoints[h].size++;
                    else
                    {
                        auto it =  m_gridNumPoints.find(h);
                        if(it == m_gridNumPoints.end())
                        {
                            m_gridNumPoints[h].size = 0;
                            Vertexf center(
                                    (idx + dx) * this->m_volumen + m_bb.getMin()[0],
                                    (idx + dy) * this->m_volumen + m_bb.getMin()[1],
                                    (idx + dz) * this->m_volumen + m_bb.getMin()[2]);
                            Vertexf min(center[0]-(m_volumen/2),center[1]-(m_volumen/2),center[2]-(m_volumen/2) );
                            Vertexf max(center[0]+(m_volumen/2),center[1]+(m_volumen/2),center[2]+(m_volumen/2) );
                            m_gridNumPoints[h].bb.expand(min);
                            m_gridNumPoints[h].bb.expand(max);
                        }
                    }

                }
            }
        }
        else
        {
            exit(-1);
        }
    }

    size_t num_cells = 0;
    size_t offset = 0;
    for(auto it = m_gridNumPoints.begin(); it != m_gridNumPoints.end() ; ++it)
    {
        it->second.offset = offset;
        offset+=it->second.size;
        it->second.dist_offset = num_cells++;
    }

    lineReader.rewind();

    boost::iostreams::mapped_file_params mmfparam;
    mmfparam.path = "points.mmf";
    mmfparam.mode = std::ios_base::in | std::ios_base::out | std::ios_base::trunc;
    mmfparam.new_file_size = sizeof(float)*m_numPoints*3;

    boost::iostreams::mapped_file_params mmfparam_normal;
    mmfparam_normal.path = "normals.mmf";
    mmfparam_normal.mode = std::ios_base::in | std::ios_base::out | std::ios_base::trunc;
    mmfparam_normal.new_file_size = sizeof(float)*m_numPoints*3;

    boost::iostreams::mapped_file_params mmfparam_color;
    mmfparam_color.path = "colors.mmf";
    mmfparam_color.mode = std::ios_base::in | std::ios_base::out | std::ios_base::trunc;
    mmfparam_color.new_file_size = sizeof(unsigned char)*m_numPoints*3;

    m_PointFile.open(mmfparam);
    float *mmfdata_normal;
    unsigned char *mmfdata_color;
    if(lineReader.getFileType() == XYZNRGB || lineReader.getFileType() == XYZN)
    {
        m_NomralFile.open(mmfparam_normal);
        mmfdata_normal = (float*)m_NomralFile.data();
        m_has_normal = true;
    }
    if(lineReader.getFileType() == XYZNRGB || lineReader.getFileType() == XYZRGB)
    {
        m_ColorFile.open(mmfparam_color);
        mmfdata_color =  (unsigned char*)m_ColorFile.data();
        m_has_color = true;
    }
    float * mmfdata = (float*)m_PointFile.data();

    while (true) {
        if (lineReader.getFileType() == XYZNRGB) {
            boost::shared_ptr<xyznc> a = boost::static_pointer_cast<xyznc>(lineReader.getNextPoints(rsize,1024));
            if (rsize <= 0) {
                break;
            }
            for (int i = 0; i < rsize; i++) {
                ix = a.get()[i].point.x*m_scale;
                iy = a.get()[i].point.y*m_scale;
                iz = a.get()[i].point.z*m_scale;
                size_t idx = calcIndex((ix - m_bb.getMin()[0])/volumen);
                size_t idy = calcIndex((iy - m_bb.getMin()[1])/volumen);
                size_t idz = calcIndex((iz - m_bb.getMin()[2])/volumen);
                size_t h = hashValue(idx,idy,idz);
                size_t ins = (m_gridNumPoints[h].inserted);
                m_gridNumPoints[h].ix = idx;
                m_gridNumPoints[h].iy = idy;
                m_gridNumPoints[h].iz = idz;
                m_gridNumPoints[h].inserted++;
                size_t index = m_gridNumPoints[h].offset + ins;
                mmfdata[index*3] = ix;
                mmfdata[index*3+1] = iy;
                mmfdata[index*3+2] = iz;
                mmfdata_normal[index*3] = a.get()[i].normal.x;
                mmfdata_normal[index*3+1] = a.get()[i].normal.y;
                mmfdata_normal[index*3+2] = a.get()[i].normal.z;

                mmfdata_color[index*3] = a.get()[i].color.r;
                mmfdata_color[index*3+1] = a.get()[i].color.g;
                mmfdata_color[index*3+2] = a.get()[i].color.b;
            }
        } else if (lineReader.getFileType() == XYZN) {
            boost::shared_ptr<xyzn> a = boost::static_pointer_cast<xyzn>(lineReader.getNextPoints(rsize,1024));
            if (rsize <= 0) {
                break;
            }
            for (int i = 0; i < rsize; i++) {
                ix = a.get()[i].point.x*m_scale;
                iy = a.get()[i].point.y*m_scale;
                iz = a.get()[i].point.z*m_scale;
                size_t idx = calcIndex((ix - m_bb.getMin()[0])/volumen);
                size_t idy = calcIndex((iy - m_bb.getMin()[1])/volumen);
                size_t idz = calcIndex((iz - m_bb.getMin()[2])/volumen);
                size_t h = hashValue(idx,idy,idz);
                size_t ins = (m_gridNumPoints[h].inserted);
                m_gridNumPoints[h].ix = idx;
                m_gridNumPoints[h].iy = idy;
                m_gridNumPoints[h].iz = idz;
                m_gridNumPoints[h].inserted++;
                size_t index = m_gridNumPoints[h].offset + ins;
                mmfdata[index*3] = ix;
                mmfdata[index*3+1] = iy;
                mmfdata[index*3+2] = iz;
                mmfdata_normal[index*3] = a.get()[i].normal.x;
                mmfdata_normal[index*3+1] = a.get()[i].normal.y;
                mmfdata_normal[index*3+2] = a.get()[i].normal.z;
            }
        } else if (lineReader.getFileType() == XYZ) {
            boost::shared_ptr<xyz> a = boost::static_pointer_cast<xyz>(lineReader.getNextPoints(rsize,1024));
            if (rsize <= 0) {
                break;
            }
            for (int i = 0; i < rsize; i++) {
                ix = a.get()[i].point.x*m_scale;
                iy = a.get()[i].point.y*m_scale;
                iz = a.get()[i].point.z*m_scale;
                size_t idx = calcIndex((ix - m_bb.getMin()[0])/volumen);
                size_t idy = calcIndex((iy - m_bb.getMin()[1])/volumen);
                size_t idz = calcIndex((iz - m_bb.getMin()[2])/volumen);
                size_t h = hashValue(idx,idy,idz);
                size_t ins = (m_gridNumPoints[h].inserted);
                m_gridNumPoints[h].ix = idx;
                m_gridNumPoints[h].iy = idy;
                m_gridNumPoints[h].iz = idz;
                m_gridNumPoints[h].inserted++;
                size_t index = m_gridNumPoints[h].offset + ins;
                mmfdata[index*3] = ix;
                mmfdata[index*3+1] = iy;
                mmfdata[index*3+2] = iz;
            }
        }
        else if (lineReader.getFileType() == XYZRGB) {
            boost::shared_ptr<xyzc> a = boost::static_pointer_cast<xyzc>(lineReader.getNextPoints(rsize,1024));
            if (rsize <= 0) {
                break;
            }
            for (int i = 0; i < rsize; i++) {
                ix = a.get()[i].point.x*m_scale;
                iy = a.get()[i].point.y*m_scale;
                iz = a.get()[i].point.z*m_scale;
                size_t idx = calcIndex((ix - m_bb.getMin()[0])/volumen);
                size_t idy = calcIndex((iy - m_bb.getMin()[1])/volumen);
                size_t idz = calcIndex((iz - m_bb.getMin()[2])/volumen);
                size_t h = hashValue(idx,idy,idz);
                size_t ins = (m_gridNumPoints[h].inserted);
                m_gridNumPoints[h].ix = idx;
                m_gridNumPoints[h].iy = idy;
                m_gridNumPoints[h].iz = idz;
                m_gridNumPoints[h].inserted++;
                size_t index = m_gridNumPoints[h].offset + ins;
                mmfdata[index*3] = ix;
                mmfdata[index*3+1] = iy;
                mmfdata[index*3+2] = iz;
                mmfdata_color[index*3] = a.get()[i].color.r;
                mmfdata_color[index*3+1] = a.get()[i].color.g;
                mmfdata_color[index*3+2] = a.get()[i].color.b;
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
    mmfparam.new_file_size = sizeof(float)*size()*8;

    m_PointFile.open(mmfparam);
    m_PointFile.close();
    size_t count = 0;
    for(auto it = m_gridNumPoints.begin(); it!=m_gridNumPoints.end(); it++)
    {
        grids.push_back(it->first);
    }
}
lvr::floatArr VolumenGrid::points(size_t i, size_t& numPoints)
{
    numPoints = m_gridNumPoints[grids[i]].size;
    lvr::floatArr points(new float[numPoints*3]);


    boost::iostreams::mapped_file_source mmfs("points.mmf");
    float * mmfdata = (float*)mmfs.data();
    size_t p_index = 0;
    for(size_t x = 0 ; x <  numPoints; x++)
    {
        points.get()[p_index] = mmfdata[( m_gridNumPoints[grids[i]].offset+x)*3];
        points.get()[p_index+1] = mmfdata[( m_gridNumPoints[grids[i]].offset+x)*3+1];
        points.get()[p_index+2] = mmfdata[( m_gridNumPoints[grids[i]].offset+x)*3+2];
        p_index+=3;
    }
    return points;
}
