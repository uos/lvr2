//
// Created by imitschke on 17.07.17.
//

#include "BigVolumen.hpp"
#include "lvr/io/Timestamp.hpp"
#include <cstring>
#include "LineReader.hpp"
#include <lvr/reconstruction/FastReconstructionTables.hpp>
BigVolumen::BigVolumen(std::vector<std::string> cloudPath, float voxelsize, float overlapping_size, float scale) :
        m_maxIndex(0), m_maxIndexSquare(0), m_maxIndexX(0), m_maxIndexY(0), m_maxIndexZ(0), m_numPoints(0), m_extrude(true),m_scale(scale),
        m_has_normal(false), m_has_color(false)
{
    omp_init_lock(&m_lock);
    if(overlapping_size==0) overlapping_size = voxelsize/10;
    m_voxelSize = voxelsize;
    float overlapp_size = overlapping_size;
    //First, parse whole file to get BoundingBox and amount of points
    float ix,iy,iz;
    std::cout << lvr::timestamp << " Starting BB" << std::endl;
    m_numPoints = 0;
    size_t rsize = 0;
    LineReader lineReader(cloudPath);
    size_t lasti = 0;
    while(lineReader.ok())
    {
        if(lineReader.getFileType() == XYZNRGB)
        {
            boost::shared_ptr<xyznc> a = boost::static_pointer_cast<xyznc> (lineReader.getNextPoints(rsize,1024));
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
            boost::shared_ptr<xyzn> a = boost::static_pointer_cast<xyzn> (lineReader.getNextPoints(rsize,1024));
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
            boost::shared_ptr<xyz> a = boost::static_pointer_cast<xyz> (lineReader.getNextPoints(rsize,1024));
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
            boost::shared_ptr<xyzc> a = boost::static_pointer_cast<xyzc> (lineReader.getNextPoints(rsize,1024));
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


    //
    lineReader.rewind();

    for(auto it = m_gridNumPoints.begin(); it != m_gridNumPoints.end(); it++)
    {
        float maxx = m_bb.getMin().x + it->second.ix * m_voxelSize;
        float maxy = m_bb.getMin().y + it->second.iy * m_voxelSize;
        float maxz = m_bb.getMin().z + it->second.iz * m_voxelSize;

        float minx = m_bb.getMin().x + it->second.ix * (m_voxelSize*2);
        float miny = m_bb.getMin().y + it->second.iy * (m_voxelSize*2);
        float minz = m_bb.getMin().z + it->second.iz * (m_voxelSize*2);

        it->second.bb.expand(maxx, maxy, maxz);
        it->second.bb.expand(minx, miny, minz);
    }

    size_t idx, idy, idz;

    if(lineReader.getFileType() == XYZNRGB || lineReader.getFileType() == XYZN)
    {
        m_has_normal = true;
    }
    if(lineReader.getFileType() == XYZNRGB || lineReader.getFileType() == XYZRGB)
    {
        m_has_color = true;
    }

    while (lineReader.ok())
    {
        if (lineReader.getFileType() == XYZNRGB)
        {
            boost::shared_ptr<xyznc> a = boost::static_pointer_cast<xyznc>(lineReader.getNextPoints(rsize,1024));
            if (rsize <= 0  && !lineReader.ok())
            {
                break;
            }
            for (int i = 0; i < rsize; i++)
            {
                ix = a.get()[i].point.x*m_scale;
                iy = a.get()[i].point.y*m_scale;
                iz = a.get()[i].point.z*m_scale;
                size_t idx = calcIndex((ix - m_bb.getMin()[0])/voxelsize);
                size_t idy = calcIndex((iy - m_bb.getMin()[1])/voxelsize);
                size_t idz = calcIndex((iz - m_bb.getMin()[2])/voxelsize);
                size_t h = hashValue(idx,idy,idz);
                m_gridNumPoints[h].ix = idx;
                m_gridNumPoints[h].iy = idy;
                m_gridNumPoints[h].iz = idz;
                m_gridNumPoints[h].size++;
                if(!m_gridNumPoints[h].ofs_points.is_open())
                {
                    std::stringstream ss;
                    ss << "part-" << idx << "-" << idy << "-" << idz << "-points.binary";
                    m_gridNumPoints[h].ofs_points.open(ss.str(), std::ofstream::binary);
                }
                m_gridNumPoints[h].ofs_points.write((char*)&ix,sizeof(float));
                m_gridNumPoints[h].ofs_points.write((char*)&iy,sizeof(float));
                m_gridNumPoints[h].ofs_points.write((char*)&iz,sizeof(float));

                if(!m_gridNumPoints[h].ofs_normals.is_open())
                {
                    std::stringstream ss;
                    ss << "part-" << idx << "-" << idy << "-" << idz << "-normals.binary";
                    m_gridNumPoints[h].ofs_normals.open(ss.str(), std::ofstream::binary);
                }
                m_gridNumPoints[h].ofs_normals.write((char*)&a.get()[i].normal.x,sizeof(float));
                m_gridNumPoints[h].ofs_normals.write((char*)&a.get()[i].normal.y,sizeof(float));
                m_gridNumPoints[h].ofs_normals.write((char*)&a.get()[i].normal.z,sizeof(float));

                if(!m_gridNumPoints[h].ofs_colors.is_open())
                {
                    std::stringstream ss;
                    ss << "part-" << idx << "-" << idy << "-" << idz << "-colors.binary";
                    m_gridNumPoints[h].ofs_colors.open(ss.str(), std::ofstream::binary);
                }
                m_gridNumPoints[h].ofs_colors.write((char*)&a.get()[i].color.r,sizeof(unsigned char));
                m_gridNumPoints[h].ofs_colors.write((char*)&a.get()[i].color.g,sizeof(unsigned char));
                m_gridNumPoints[h].ofs_colors.write((char*)&a.get()[i].color.b,sizeof(unsigned char));
            }
        } else if (lineReader.getFileType() == XYZN) {
            boost::shared_ptr<xyzn> a = boost::static_pointer_cast<xyzn>(lineReader.getNextPoints(rsize,1024));
            if (rsize <= 0  && !lineReader.ok())
            {
                break;
            }
            for (int i = 0; i < rsize; i++)
            {
                ix = a.get()[i].point.x*m_scale;
                iy = a.get()[i].point.y*m_scale;
                iz = a.get()[i].point.z*m_scale;
                size_t idx = calcIndex((ix - m_bb.getMin()[0])/voxelsize);
                size_t idy = calcIndex((iy - m_bb.getMin()[1])/voxelsize);
                size_t idz = calcIndex((iz - m_bb.getMin()[2])/voxelsize);
                size_t h = hashValue(idx,idy,idz);
                m_gridNumPoints[h].ix = idx;
                m_gridNumPoints[h].iy = idy;
                m_gridNumPoints[h].iz = idz;
                m_gridNumPoints[h].size++;
                if(!m_gridNumPoints[h].ofs_points.is_open())
                {
                    std::stringstream ss;
                    ss << "part-" << idx << "-" << idy << "-" << idz << "-points.binary";
                    m_gridNumPoints[h].ofs_points.open(ss.str(), std::ofstream::binary);
                }
                m_gridNumPoints[h].ofs_points.write((char*)&ix,sizeof(float));
                m_gridNumPoints[h].ofs_points.write((char*)&iy,sizeof(float));
                m_gridNumPoints[h].ofs_points.write((char*)&iz,sizeof(float));

                if(!m_gridNumPoints[h].ofs_normals.is_open())
                {
                    std::stringstream ss;
                    ss << "part-" << idx << "-" << idy << "-" << idz << "-normals.binary";
                    m_gridNumPoints[h].ofs_normals.open(ss.str(), std::ofstream::binary);
                }
                m_gridNumPoints[h].ofs_normals.write((char*)&a.get()[i].normal.x,sizeof(float));
                m_gridNumPoints[h].ofs_normals.write((char*)&a.get()[i].normal.y,sizeof(float));
                m_gridNumPoints[h].ofs_normals.write((char*)&a.get()[i].normal.z,sizeof(float));
            }
        } else if (lineReader.getFileType() == XYZ)
        {
            boost::shared_ptr<xyz> a = boost::static_pointer_cast<xyz>(lineReader.getNextPoints(rsize,1024));
            if (rsize <= 0  && !lineReader.ok())
            {
                break;
            }
            for (int i = 0; i < rsize; i++)
            {
                ix = a.get()[i].point.x*m_scale;
                iy = a.get()[i].point.y*m_scale;
                iz = a.get()[i].point.z*m_scale;
                size_t idx = calcIndex((ix - m_bb.getMin()[0])/voxelsize);
                size_t idy = calcIndex((iy - m_bb.getMin()[1])/voxelsize);
                size_t idz = calcIndex((iz - m_bb.getMin()[2])/voxelsize);
                size_t h = hashValue(idx,idy,idz);
                m_gridNumPoints[h].ix = idx;
                m_gridNumPoints[h].iy = idy;
                m_gridNumPoints[h].iz = idz;
                m_gridNumPoints[h].size++;
                if(!m_gridNumPoints[h].ofs_points.is_open())
                {
                    std::stringstream ss;
                    ss << "part-" << idx << "-" << idy << "-" << idz << "-points.binary";
                    m_gridNumPoints[h].ofs_points.open(ss.str(), std::ofstream::binary);
                }
                m_gridNumPoints[h].ofs_points.write((char*)&ix,sizeof(float));
                m_gridNumPoints[h].ofs_points.write((char*)&iy,sizeof(float));
                m_gridNumPoints[h].ofs_points.write((char*)&iz,sizeof(float));
            }
        }
        else if (lineReader.getFileType() == XYZRGB)
        {
            boost::shared_ptr<xyzc> a = boost::static_pointer_cast<xyzc>(lineReader.getNextPoints(rsize,1024));
            if (rsize <= 0  && !lineReader.ok())
            {
                break;
            }
            for (int i = 0; i < rsize; i++)
            {
                ix = a.get()[i].point.x*m_scale;
                iy = a.get()[i].point.y*m_scale;
                iz = a.get()[i].point.z*m_scale;
                size_t idx = calcIndex((ix - m_bb.getMin()[0])/voxelsize);
                size_t idy = calcIndex((iy - m_bb.getMin()[1])/voxelsize);
                size_t idz = calcIndex((iz - m_bb.getMin()[2])/voxelsize);
                size_t h = hashValue(idx,idy,idz);
                m_gridNumPoints[h].ix = idx;
                m_gridNumPoints[h].iy = idy;
                m_gridNumPoints[h].iz = idz;
                m_gridNumPoints[h].size++;
                if(!m_gridNumPoints[h].ofs_points.is_open())
                {
                    std::stringstream ss;
                    ss << "part-" << idx << "-" << idy << "-" << idz << "-points.binary";
                    m_gridNumPoints[h].ofs_points.open(ss.str(), std::ofstream::binary);
                }
                m_gridNumPoints[h].ofs_points.write((char*)&ix,sizeof(float));
                m_gridNumPoints[h].ofs_points.write((char*)&iy,sizeof(float));
                m_gridNumPoints[h].ofs_points.write((char*)&iz,sizeof(float));

                if(!m_gridNumPoints[h].ofs_colors.is_open())
                {
                    std::stringstream ss;
                    ss << "part-" << idx << "-" << idy << "-" << idz << "-colors.binary";
                    m_gridNumPoints[h].ofs_colors.open(ss.str(), std::ofstream::binary);
                }
                m_gridNumPoints[h].ofs_colors.write((char*)&a.get()[i].color.r,sizeof(unsigned char));
                m_gridNumPoints[h].ofs_colors.write((char*)&a.get()[i].color.g,sizeof(unsigned char));
                m_gridNumPoints[h].ofs_colors.write((char*)&a.get()[i].color.b,sizeof(unsigned char));
            }
        }
    }



    // Add overlapping points
    for(auto cell = m_gridNumPoints.begin() ; cell != m_gridNumPoints.end(); cell++)
    { 
        if (lineReader.getFileType() == XYZ)
        {
            stringstream ss_points;
            ss_points << "part-" << cell->second.ix << "-" << cell->second.iy << "-" << cell->second.iz << "-points.binary";
            ifstream ifs_points(ss_points.str(), std::ifstream::binary);
            lvr::floatArr pointBuffer(new float[cell->second.size*3]);
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
                        neigbout_it->second.ofs_points.write((char*)&x,sizeof(float));
                        neigbout_it->second.ofs_points.write((char*)&y,sizeof(float));
                        neigbout_it->second.ofs_points.write((char*)&z,sizeof(float));
                    }
                }
            }
        }
        else if (lineReader.getFileType() == XYZNRGB)
        {
            stringstream ss_points;
            ss_points << "part-" << cell->second.ix << "-" << cell->second.iy << "-" << cell->second.iz << "-points.binary";
            ifstream ifs_points(ss_points.str(), std::ifstream::binary);
            lvr::floatArr pointBuffer(new float[cell->second.size*3]);
            ifs_points.read((char*)pointBuffer.get(), sizeof(float)*3*cell->second.size);

            stringstream ss_normals;
            ss_normals << "part-" << cell->second.ix << "-" << cell->second.iy << "-" << cell->second.iz << "-normals.binary";
            ifstream ifs_normals(ss_normals.str(), std::ifstream::binary);
            lvr::floatArr normalBuffer(new float[cell->second.size*3]);
            ifs_normals.read((char*)normalBuffer.get(), sizeof(float)*3*cell->second.size);

            stringstream ss_colors;
            ss_colors << "part-" << cell->second.ix << "-" << cell->second.iy << "-" << cell->second.iz << "-colors.binary";
            ifstream ifs_colors(ss_colors.str(), std::ifstream::binary);
            lvr::ucharArr colorBuffer(new unsigned char[cell->second.size*3]);
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
                        neigbout_it->second.ofs_points.write((char*)&x,sizeof(float));
                        neigbout_it->second.ofs_points.write((char*)&y,sizeof(float));
                        neigbout_it->second.ofs_points.write((char*)&z,sizeof(float));

                        neigbout_it->second.ofs_normals.write((char*)&normalBuffer[i*3],sizeof(float));
                        neigbout_it->second.ofs_normals.write((char*)&normalBuffer[i*3+1],sizeof(float));
                        neigbout_it->second.ofs_normals.write((char*)&normalBuffer[i*3+2],sizeof(float));

                        neigbout_it->second.ofs_colors.write((char*)&colorBuffer[i*3],sizeof(unsigned char));
                        neigbout_it->second.ofs_colors.write((char*)&colorBuffer[i*3+1],sizeof(unsigned char));
                        neigbout_it->second.ofs_colors.write((char*)&colorBuffer[i*3+2],sizeof(unsigned char));
                    }
                }
            }
        }
        else if (lineReader.getFileType() == XYZN)
        {
            stringstream ss_points;
            ss_points << "part-" << cell->second.ix << "-" << cell->second.iy << "-" << cell->second.iz << "-points.binary";
            ifstream ifs_points(ss_points.str(), std::ifstream::binary);
            lvr::floatArr pointBuffer(new float[cell->second.size*3]);
            ifs_points.read((char*)pointBuffer.get(), sizeof(float)*3*cell->second.size);

            stringstream ss_normals;
            ss_normals << "part-" << cell->second.ix << "-" << cell->second.iy << "-" << cell->second.iz << "-normals.binary";
            ifstream ifs_normals(ss_normals.str(), std::ifstream::binary);
            lvr::floatArr normalBuffer(new float[cell->second.size*3]);
            ifs_normals.read((char*)normalBuffer.get(), sizeof(float)*3*cell->second.size);



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
                        neigbout_it->second.ofs_points.write((char*)&x,sizeof(float));
                        neigbout_it->second.ofs_points.write((char*)&y,sizeof(float));
                        neigbout_it->second.ofs_points.write((char*)&z,sizeof(float));

                        neigbout_it->second.ofs_normals.write((char*)&normalBuffer[i*3],sizeof(float));
                        neigbout_it->second.ofs_normals.write((char*)&normalBuffer[i*3+1],sizeof(float));
                        neigbout_it->second.ofs_normals.write((char*)&normalBuffer[i*3+2],sizeof(float));
                    }
                }
            }
        }
        else if (lineReader.getFileType() == XYZRGB)
        {
            stringstream ss_points;
            ss_points << "part-" << cell->second.ix << "-" << cell->second.iy << "-" << cell->second.iz << "-points.binary";
            ifstream ifs_points(ss_points.str(), std::ifstream::binary);
            lvr::floatArr pointBuffer(new float[cell->second.size*3]);
            ifs_points.read((char*)pointBuffer.get(), sizeof(float)*3*cell->second.size);

            stringstream ss_colors;
            ss_colors << "part-" << cell->second.ix << "-" << cell->second.iy << "-" << cell->second.iz << "-colors.binary";
            ifstream ifs_colors(ss_colors.str(), std::ifstream::binary);
            lvr::ucharArr colorBuffer(new unsigned char[cell->second.size*3]);
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
                        neigbout_it->second.ofs_points.write((char*)&x,sizeof(float));
                        neigbout_it->second.ofs_points.write((char*)&y,sizeof(float));
                        neigbout_it->second.ofs_points.write((char*)&z,sizeof(float));

                        neigbout_it->second.ofs_colors.write((char*)&colorBuffer[i*3],sizeof(unsigned char));
                        neigbout_it->second.ofs_colors.write((char*)&colorBuffer[i*3+1],sizeof(unsigned char));
                        neigbout_it->second.ofs_colors.write((char*)&colorBuffer[i*3+2],sizeof(unsigned char));
                    }
                }
            }
        }
    }

    for(auto cell = m_gridNumPoints.begin() ; cell != m_gridNumPoints.end(); cell++)
    {
        if(cell->second.ofs_points.is_open()) cell->second.ofs_points.close();
        if(cell->second.ofs_normals.is_open()) cell->second.ofs_normals.close();
        if(cell->second.ofs_colors.is_open()) cell->second.ofs_colors.close();
    }

}

BigVolumen::~BigVolumen() {
    omp_destroy_lock(&m_lock);
}

size_t BigVolumen::size()
{
    return m_gridNumPoints.size();
}
size_t BigVolumen::pointSize()
{
    return m_numPoints;
}










