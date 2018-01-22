//
// Created by imitschke on 17.07.17.
//

#include "BigGrid.hpp"
#include "lvr/io/Timestamp.hpp"
#include "lvr/io/Progress.hpp"
#include <cstring>
#include "LineReader.hpp"
#include <lvr/reconstruction/FastReconstructionTables.hpp>
#include <fstream>

BigGrid::BigGrid(std::vector<std::string> cloudPath, float voxelsize, float scale, size_t bufferSize) :
        m_maxIndex(0), m_maxIndexSquare(0), m_maxIndexX(0), m_maxIndexY(0), m_maxIndexZ(0), m_numPoints(0), m_extrude(true),m_scale(scale),
        m_has_normal(false), m_has_color(false), m_pointBufferSize(1024)
{
    omp_init_lock(&m_lock);
    m_voxelSize = voxelsize;
    //First, parse whole file to get BoundingBox and amount of points
    float ix,iy,iz;
    std::cout << lvr::timestamp << "Computing Bounding Box..." << std::endl;
    m_numPoints = 0;
    size_t rsize = 0;
    LineReader lineReader(cloudPath);
    size_t lasti = 0;
    while(lineReader.ok())
    {
        if(lineReader.getFileType() == XYZNRGB)
        {
            boost::shared_ptr<xyznc> a = boost::static_pointer_cast<xyznc> (lineReader.getNextPoints(rsize,m_pointBufferSize));
            if (rsize <= 0  && !lineReader.ok())
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
            boost::shared_ptr<xyzn> a = boost::static_pointer_cast<xyzn> (lineReader.getNextPoints(rsize,m_pointBufferSize));
            if (rsize <= 0  && !lineReader.ok())
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
            boost::shared_ptr<xyz> a = boost::static_pointer_cast<xyz> (lineReader.getNextPoints(rsize,m_pointBufferSize));
            if (rsize <= 0  && !lineReader.ok())
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
            boost::shared_ptr<xyzc> a = boost::static_pointer_cast<xyzc> (lineReader.getNextPoints(rsize,m_pointBufferSize));
            if (rsize <= 0  && !lineReader.ok())
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
    size_t xsize2 = calcIndex(m_bb.getXSize ()/m_voxelSize);
    float xsize = ceil(m_bb.getXSize () / voxelsize)*voxelsize;
    float ysize = ceil(m_bb.getYSize () / voxelsize)*voxelsize;
    float zsize = ceil(m_bb.getZSize () / voxelsize)*voxelsize;
    m_bb.expand(center.x + xsize/2, center.y + ysize/2, center.z + zsize/2  );
    m_bb.expand(center.x - xsize/2, center.y - ysize/2, center.z - zsize/2  );
    longestSide = ceil(longestSide / voxelsize) * voxelsize;

    //calc max indices

    //m_maxIndex = (size_t)(longestSide/voxelsize);
    m_maxIndexX = (size_t)(xsize/voxelsize);
    m_maxIndexY = (size_t)(ysize/voxelsize);
    m_maxIndexZ = (size_t)(zsize/voxelsize);
    m_maxIndex = std::max(m_maxIndexX, std::max(m_maxIndexY,m_maxIndexZ)) + 5*voxelsize ;
    m_maxIndexX+=1;
    m_maxIndexY+=2;
    m_maxIndexZ+=3;
    m_maxIndexSquare = m_maxIndex * m_maxIndex;
    std::cout << "BG: " << m_maxIndexSquare << "|" << m_maxIndexX << "|" << m_maxIndexY << "|" << m_maxIndexZ << std::endl;


    string comment = lvr::timestamp.getElapsedTime() + "Building grid... ";
    lvr::ProgressBar progress(this->m_numPoints, comment);

    //
    lineReader.rewind();

    size_t idx, idy, idz;
    while (lineReader.ok()) {
        if (lineReader.getFileType() == XYZNRGB) {
            boost::shared_ptr<xyznc> a = boost::static_pointer_cast<xyznc>(lineReader.getNextPoints(rsize,m_pointBufferSize));
            if (rsize <= 0  && !lineReader.ok())
            {
                break;
            }
            int dx, dy, dz;
            for (int i = 0; i < rsize; i++) {
                ix = a.get()[i].point.x*m_scale;
                iy = a.get()[i].point.y*m_scale;
                iz = a.get()[i].point.z*m_scale;
                 idx = calcIndex((ix - m_bb.getMin()[0])/voxelsize);
                 idy = calcIndex((iy - m_bb.getMin()[1])/voxelsize);
                 idz = calcIndex((iz - m_bb.getMin()[2])/voxelsize);
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

                        }
                    }

                }
            }
        } else if (lineReader.getFileType() == XYZN) {
            boost::shared_ptr<xyzn> a = boost::static_pointer_cast<xyzn>(lineReader.getNextPoints(rsize,m_pointBufferSize));
            if (rsize <= 0  && !lineReader.ok()){
                break;
            }
            int dx, dy, dz;
            for (int i = 0; i < rsize; i++) {
                ix = a.get()[i].point.x*m_scale;
                iy = a.get()[i].point.y*m_scale;
                iz = a.get()[i].point.z*m_scale;
                 idx = calcIndex((ix - m_bb.getMin()[0])/voxelsize);
                 idy = calcIndex((iy - m_bb.getMin()[1])/voxelsize);
                 idz = calcIndex((iz - m_bb.getMin()[2])/voxelsize);

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

                        }
                    }

                }



            }
        } else if (lineReader.getFileType() == XYZ) {
            boost::shared_ptr<xyz> a = boost::static_pointer_cast<xyz>(lineReader.getNextPoints(rsize,m_pointBufferSize));
            if (rsize <= 0  && !lineReader.ok()){
                break;
            }
            int dx, dy, dz;
            for (int i = 0; i < rsize; i++) {
                ix = a.get()[i].point.x*m_scale;
                iy = a.get()[i].point.y*m_scale;
                iz = a.get()[i].point.z*m_scale;
                 idx = calcIndex((ix - m_bb.getMin()[0])/voxelsize);
                 idy = calcIndex((iy - m_bb.getMin()[1])/voxelsize);
                 idz = calcIndex((iz - m_bb.getMin()[2])/voxelsize);
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

                        }
                    }

                }
            }
        }
        else if (lineReader.getFileType() == XYZRGB) {
            boost::shared_ptr<xyzc> a = boost::static_pointer_cast<xyzc>(lineReader.getNextPoints(rsize,m_pointBufferSize));
            if (rsize <= 0  && !lineReader.ok()){
                break;
            }
            int dx, dy, dz;
            for (int i = 0; i < rsize; i++) {
                ix = a.get()[i].point.x*m_scale;
                iy = a.get()[i].point.y*m_scale;
                iz = a.get()[i].point.z*m_scale;
                 idx = calcIndex((ix - m_bb.getMin()[0])/voxelsize);
                 idy = calcIndex((iy - m_bb.getMin()[1])/voxelsize);
                 idz = calcIndex((iz - m_bb.getMin()[2])/voxelsize);
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

                        }
                    }

                }
            }
        }
        else
        {
            exit(-1);
        }
        progress+= rsize;
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

    while (lineReader.ok()) {
        if (lineReader.getFileType() == XYZNRGB) {
            boost::shared_ptr<xyznc> a = boost::static_pointer_cast<xyznc>(lineReader.getNextPoints(rsize,m_pointBufferSize));
            if (rsize <= 0  && !lineReader.ok()){
                break;
            }
            for (int i = 0; i < rsize; i++) {
                ix = a.get()[i].point.x*m_scale;
                iy = a.get()[i].point.y*m_scale;
                iz = a.get()[i].point.z*m_scale;
                size_t idx = calcIndex((ix - m_bb.getMin()[0])/voxelsize);
                size_t idy = calcIndex((iy - m_bb.getMin()[1])/voxelsize);
                size_t idz = calcIndex((iz - m_bb.getMin()[2])/voxelsize);
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
            boost::shared_ptr<xyzn> a = boost::static_pointer_cast<xyzn>(lineReader.getNextPoints(rsize,m_pointBufferSize));
            if (rsize <= 0  && !lineReader.ok()){
                break;
            }
            for (int i = 0; i < rsize; i++) {
                ix = a.get()[i].point.x*m_scale;
                iy = a.get()[i].point.y*m_scale;
                iz = a.get()[i].point.z*m_scale;
                size_t idx = calcIndex((ix - m_bb.getMin()[0])/voxelsize);
                size_t idy = calcIndex((iy - m_bb.getMin()[1])/voxelsize);
                size_t idz = calcIndex((iz - m_bb.getMin()[2])/voxelsize);
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
            boost::shared_ptr<xyz> a = boost::static_pointer_cast<xyz>(lineReader.getNextPoints(rsize,m_pointBufferSize));
            if (rsize <= 0  && !lineReader.ok()){
                break;
            }
            for (int i = 0; i < rsize; i++) {
                ix = a.get()[i].point.x*m_scale;
                iy = a.get()[i].point.y*m_scale;
                iz = a.get()[i].point.z*m_scale;
                size_t idx = calcIndex((ix - m_bb.getMin()[0])/voxelsize);
                size_t idy = calcIndex((iy - m_bb.getMin()[1])/voxelsize);
                size_t idz = calcIndex((iz - m_bb.getMin()[2])/voxelsize);
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
            boost::shared_ptr<xyzc> a = boost::static_pointer_cast<xyzc>(lineReader.getNextPoints(rsize,m_pointBufferSize));
            if (rsize <= 0  && !lineReader.ok()){
                break;
            }
            for (int i = 0; i < rsize; i++) {
                ix = a.get()[i].point.x*m_scale;
                iy = a.get()[i].point.y*m_scale;
                iz = a.get()[i].point.z*m_scale;
                size_t idx = calcIndex((ix - m_bb.getMin()[0])/voxelsize);
                size_t idy = calcIndex((iy - m_bb.getMin()[1])/voxelsize);
                size_t idz = calcIndex((iz - m_bb.getMin()[2])/voxelsize);
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


}

BigGrid::~BigGrid() {
    omp_destroy_lock(&m_lock);
}

BigGrid::BigGrid(std::string path)
{
    ifstream ifs(path,ios::binary);

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
    float mx,my,mz,n1,n2,n3;
    ifs.read((char*)&mx, sizeof(float));
    ifs.read((char*)&my, sizeof(float));
    ifs.read((char*)&mz, sizeof(float));
    ifs.read((char*)&n1, sizeof(float));
    ifs.read((char*)&n2, sizeof(float));
    ifs.read((char*)&n3, sizeof(float));
    m_bb.expand(mx,my,mz);
    m_bb.expand(n1,n2,n3);

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

    for(size_t i = 0 ; i<gridSize;i++)
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

void BigGrid::serialize(std::string path)
{
    ofstream ofs(path,ios::binary);
//    size_t data_size =      sizeof(m_maxIndexSquare) + sizeof(m_maxIndex) + sizeof(m_maxIndexX) + sizeof(m_maxIndexY) +
//                            sizeof(m_maxIndexZ) + sizeof(m_numPoints) +  sizeof(m_pointBufferSize) +
//                            sizeof(m_voxelSize) + sizeof(m_extrude) + sizeof(m_has_normal) + sizeof(m_has_color) +
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
    for(auto it = m_gridNumPoints.begin() ; it!= m_gridNumPoints.end(); ++it)
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

size_t BigGrid::size()
{
    return m_gridNumPoints.size();
}
size_t BigGrid::pointSize()
{
    return m_numPoints;
}
size_t BigGrid::pointSize(int i, int j, int k)
{
    size_t h = hashValue(i,j,k);
    auto it = m_gridNumPoints.find(h);
    if(it == m_gridNumPoints.end())
    {
        return 0;
    }
    else
    {
        return m_gridNumPoints[h].size;
    }

}
lvr::floatArr BigGrid::points(int i, int j, int k, size_t& numPoints)
{
    lvr::floatArr points;
    size_t h = hashValue(i,j,k);
    auto it = m_gridNumPoints.find(h);
    if(it != m_gridNumPoints.end())
    {
        size_t cellSize= m_gridNumPoints[h].size;

        points = lvr::floatArr(new float[3*cellSize]);
        boost::iostreams::mapped_file_params mmfparam;
        mmfparam.path = "points.mmf";
        mmfparam.mode = std::ios_base::in | std::ios_base::out | std::ios_base::trunc ;

        m_PointFile.open(mmfparam);
        float * mmfdata = (float*)m_PointFile.data();

        memcpy ( points.get(), mmfdata, 3*pointSize()*sizeof(float));

        numPoints = pointSize();
    }
    return points;

}
lvr::floatArr BigGrid::points(float minx, float miny, float minz, float maxx, float maxy, float maxz, size_t& numPoints)
{
    minx = (minx > m_bb.getMin()[0]) ? minx : m_bb.getMin()[0];
    miny = (miny > m_bb.getMin()[1]) ? miny : m_bb.getMin()[1];
    minz = (minz > m_bb.getMin()[2]) ? minz : m_bb.getMin()[2];
    maxx = (maxx < m_bb.getMax()[0]) ? maxx : m_bb.getMax()[0];
    maxy = (maxy < m_bb.getMax()[1]) ? maxy : m_bb.getMax()[1];
    maxz = (maxz < m_bb.getMax()[2]) ? maxz : m_bb.getMax()[2];

    size_t idxmin = calcIndex((minx - m_bb.getMin()[0])/m_voxelSize);
    size_t idymin = calcIndex((miny - m_bb.getMin()[1])/m_voxelSize);
    size_t idzmin = calcIndex((minz - m_bb.getMin()[2])/m_voxelSize);
    size_t idxmax = calcIndex((maxx - m_bb.getMin()[0])/m_voxelSize);
    size_t idymax = calcIndex((maxy - m_bb.getMin()[1])/m_voxelSize);
    size_t idzmax = calcIndex((maxz - m_bb.getMin()[2])/m_voxelSize);

    numPoints = getSizeofBox( minx,  miny,  minz,  maxx,  maxy,  maxz);

    lvr::floatArr points(new float[numPoints*3]);
    size_t p_index = 0;


    boost::iostreams::mapped_file_source mmfs("points.mmf");
    float * mmfdata = (float*)mmfs.data();





    for(auto it = m_gridNumPoints.begin() ; it!=m_gridNumPoints.end() ; it++)
    {
        if(
                it->second.ix >= idxmin &&
                it->second.iy >= idymin &&
                it->second.iz >= idzmin &&
                it->second.ix <= idxmax &&
                it->second.iy <= idymax &&
                it->second.iz <= idzmax
                )
        {
            size_t cSize = it->second.size;
            for(size_t x = 0 ; x <  cSize; x++)
            {
                points.get()[p_index] = mmfdata[(it->second.offset+x)*3];
                points.get()[p_index+1] = mmfdata[(it->second.offset+x)*3+1];
                points.get()[p_index+2] = mmfdata[(it->second.offset+x)*3+2];
                p_index+=3;
            }

        }
    }

    return points;
}

lvr::floatArr BigGrid::normals(float minx, float miny, float minz, float maxx, float maxy, float maxz, size_t& numPoints)
{
    std::ifstream ifs("normals.mmf");
    if(!ifs.good())
    {
        numPoints = 0;
        lvr::floatArr  arr;
        return arr;
    }
    minx = (minx > m_bb.getMin()[0]) ? minx : m_bb.getMin()[0];
    miny = (miny > m_bb.getMin()[1]) ? miny : m_bb.getMin()[1];
    minz = (minz > m_bb.getMin()[2]) ? minz : m_bb.getMin()[2];
    maxx = (maxx < m_bb.getMax()[0]) ? maxx : m_bb.getMax()[0];
    maxy = (maxy < m_bb.getMax()[1]) ? maxy : m_bb.getMax()[1];
    maxz = (maxz < m_bb.getMax()[2]) ? maxz : m_bb.getMax()[2];

    size_t idxmin = calcIndex((minx - m_bb.getMin()[0])/m_voxelSize);
    size_t idymin = calcIndex((miny - m_bb.getMin()[1])/m_voxelSize);
    size_t idzmin = calcIndex((minz - m_bb.getMin()[2])/m_voxelSize);
    size_t idxmax = calcIndex((maxx - m_bb.getMin()[0])/m_voxelSize);
    size_t idymax = calcIndex((maxy - m_bb.getMin()[1])/m_voxelSize);
    size_t idzmax = calcIndex((maxz - m_bb.getMin()[2])/m_voxelSize);

    numPoints = getSizeofBox( minx,  miny,  minz,  maxx,  maxy,  maxz);

    lvr::floatArr points(new float[numPoints*3]);
    size_t p_index = 0;


    boost::iostreams::mapped_file_source mmfs("normals.mmf");
    float * mmfdata = (float*)mmfs.data();





    for(auto it = m_gridNumPoints.begin() ; it!=m_gridNumPoints.end() ; it++)
    {
        if(
                it->second.ix >= idxmin &&
                it->second.iy >= idymin &&
                it->second.iz >= idzmin &&
                it->second.ix <= idxmax &&
                it->second.iy <= idymax &&
                it->second.iz <= idzmax
                )
        {
            size_t cSize = it->second.size;
            for(size_t x = 0 ; x <  cSize; x++)
            {
                points.get()[p_index] = mmfdata[(it->second.offset+x)*3];
                points.get()[p_index+1] = mmfdata[(it->second.offset+x)*3+1];
                points.get()[p_index+2] = mmfdata[(it->second.offset+x)*3+2];
                p_index+=3;
            }

        }
    }

    return points;
}

lvr::ucharArr BigGrid::colors(float minx, float miny, float minz, float maxx, float maxy, float maxz, size_t& numPoints)
{
    std::ifstream ifs("colors.mmf");
    if(!ifs.good())
    {
        numPoints = 0;
        lvr::ucharArr arr;
        return arr;
    }
    minx = (minx > m_bb.getMin()[0]) ? minx : m_bb.getMin()[0];
    miny = (miny > m_bb.getMin()[1]) ? miny : m_bb.getMin()[1];
    minz = (minz > m_bb.getMin()[2]) ? minz : m_bb.getMin()[2];
    maxx = (maxx < m_bb.getMax()[0]) ? maxx : m_bb.getMax()[0];
    maxy = (maxy < m_bb.getMax()[1]) ? maxy : m_bb.getMax()[1];
    maxz = (maxz < m_bb.getMax()[2]) ? maxz : m_bb.getMax()[2];

    size_t idxmin = calcIndex((minx - m_bb.getMin()[0])/m_voxelSize);
    size_t idymin = calcIndex((miny - m_bb.getMin()[1])/m_voxelSize);
    size_t idzmin = calcIndex((minz - m_bb.getMin()[2])/m_voxelSize);
    size_t idxmax = calcIndex((maxx - m_bb.getMin()[0])/m_voxelSize);
    size_t idymax = calcIndex((maxy - m_bb.getMin()[1])/m_voxelSize);
    size_t idzmax = calcIndex((maxz - m_bb.getMin()[2])/m_voxelSize);

    numPoints = getSizeofBox( minx,  miny,  minz,  maxx,  maxy,  maxz);

    lvr::ucharArr points(new unsigned char[numPoints*3]);
    size_t p_index = 0;


    boost::iostreams::mapped_file_source mmfs("colors.mmf");
    unsigned char * mmfdata = (unsigned char*)mmfs.data();





    for(auto it = m_gridNumPoints.begin() ; it!=m_gridNumPoints.end() ; it++)
    {
        if(
                it->second.ix >= idxmin &&
                it->second.iy >= idymin &&
                it->second.iz >= idzmin &&
                it->second.ix <= idxmax &&
                it->second.iy <= idymax &&
                it->second.iz <= idzmax
                )
        {
            size_t cSize = it->second.size;
            for(size_t x = 0 ; x <  cSize; x++)
            {
                points.get()[p_index] = mmfdata[(it->second.offset+x)*3];
                points.get()[p_index+1] = mmfdata[(it->second.offset+x)*3+1];
                points.get()[p_index+2] = mmfdata[(it->second.offset+x)*3+2];
                p_index+=3;
            }

        }
    }

    return points;
}

bool BigGrid::exists(int i, int j, int k)
{
    size_t h = hashValue(i,j,k);
    auto it = m_gridNumPoints.find(h);
    return it != m_gridNumPoints.end();
}
void BigGrid::insert(float x, float y, float z)
{

}

lvr::floatArr BigGrid::getPointCloud(size_t & numPoints)
{
    lvr::floatArr points(new float[3*pointSize()]);
    boost::iostreams::mapped_file_params mmfparam;
    mmfparam.path = "points.mmf";
    mmfparam.mode = std::ios_base::in | std::ios_base::out | std::ios_base::trunc;

    m_PointFile.open(mmfparam);
    float * mmfdata = (float*)m_PointFile.data();
    memcpy ( points.get(), mmfdata, 3*pointSize()*sizeof(float));

    numPoints = pointSize();
    return points;

}

size_t BigGrid::getSizeofBox(float minx, float miny, float minz, float maxx, float maxy, float maxz)
{
    size_t idxmin = calcIndex((minx - m_bb.getMin()[0])/m_voxelSize);
    size_t idymin = calcIndex((miny - m_bb.getMin()[1])/m_voxelSize);
    size_t idzmin = calcIndex((minz - m_bb.getMin()[2])/m_voxelSize);
    size_t idxmax = calcIndex((maxx - m_bb.getMin()[0])/m_voxelSize);
    size_t idymax = calcIndex((maxy - m_bb.getMin()[1])/m_voxelSize);
    size_t idzmax = calcIndex((maxz - m_bb.getMin()[2])/m_voxelSize);


    size_t numPoints = 0;

    // Overhead of saving indices needed to speedup size lookup
    for(auto it = m_gridNumPoints.begin() ; it!=m_gridNumPoints.end() ; it++)
    {
        if(
            it->second.ix >= idxmin &&
            it->second.iy >= idymin &&
            it->second.iz >= idzmin &&
            it->second.ix <= idxmax &&
            it->second.iy <= idymax &&
            it->second.iz <= idzmax
        )
        {
            numPoints+=it->second.size;
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
