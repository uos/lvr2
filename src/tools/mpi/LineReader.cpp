//
// Created by imitschke on 15.08.17.
//

#include "LineReader.hpp"
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <sstream>
#include <exception>
#include <stdio.h>
#include <memory>
#include <iostream>
LineReader::LineReader()
{

}

LineReader::LineReader(std::string filePath)
{
    open(filePath);
}

void LineReader::open(std::string filePath)
{
    m_filePath = filePath;
    bool gotxyz = false;
    bool gotcolor = false;
    bool gotnormal = false;
    bool readHeader = false;

    if(boost::algorithm::contains(filePath,".ply"))
    {
        m_ply = true;
    }
    else
    {
        m_ply = false;
    }

    std::ifstream ifs(m_filePath);

    if(m_ply)
    {
        std::string line;
        while(! readHeader)
        {
            std::getline(ifs,line);
            if(boost::algorithm::contains(line,"element point") || boost::algorithm::contains(line,"element point"))
            {
                std::stringstream ss(line);
                string tmp;
                ss >> tmp;
                ss >> tmp;
                ss >> m_elementAmount;
            }
            else if(boost::algorithm::contains(line,"property float x") ||
                    boost::algorithm::contains(line,"property float y") ||
                    boost::algorithm::contains(line,"property float z"))
            {
                gotxyz = true;
            }
            else if(boost::algorithm::contains(line,"property uchar red") ||
                    boost::algorithm::contains(line,"property uchar green") ||
                    boost::algorithm::contains(line,"property uchar blue"))
            {
                gotcolor = true;
            }
            else if(boost::algorithm::contains(line,"property float nx") ||
                    boost::algorithm::contains(line,"property float ny") ||
                    boost::algorithm::contains(line,"property float nz"))
            {
                gotnormal = true;
            }
            else if(boost::algorithm::contains(line,"end_header"))
            {
                readHeader = true;
            }
            else if(boost::algorithm::contains(line,"binary"))
            {
                m_binary = true;
            }
            else if(boost::algorithm::contains(line,"ascii"))
            {
                m_binary = false;
            }
        }
    }
    else
    {
        std::cout << "File Type is not PLY, checking file... " << std::endl;
        std::string line;
        std::getline(ifs,line);
        std::stringstream ss(line);
        string tmp;
        unsigned int number_of_line_elements = 0;
        while(ss >> tmp)
        {
            number_of_line_elements++;
            if(number_of_line_elements == 3) gotxyz = true;
            if(number_of_line_elements == 6)
            {
                if(boost::algorithm::contains(tmp,"."))
                {
                    gotnormal = true;
                }
                else
                {
                    gotcolor = true;
                }
            }
            if(number_of_line_elements == 9)
            {
                gotnormal = true;
                gotcolor = true;
            }
            if(number_of_line_elements>9)
            {
                throw std::range_error("Wrong file format, expecting file ascii or ply file format, ascii file format must have order:  x y z [nx ny nz] [cx cy cz] (points, normals, colors)");
            }
        }
        ifs.seekg(0);


    }

    m_filePos = ifs.tellg();
    if(gotxyz && gotcolor && gotnormal)
    {
        m_fileType = XYZNRGB;
        m_PointBlockSize = sizeof(float)*3 + sizeof(unsigned char) * 3 + sizeof(float)*3;
    }
    else if(gotxyz && gotcolor && !gotnormal)
    {
        m_fileType = XYZRGB;
        m_PointBlockSize = sizeof(float)*3 + sizeof(unsigned char) * 3;
    }
    else if(gotxyz && !gotcolor && gotnormal)
    {
        m_fileType = XYZN;
        m_PointBlockSize = sizeof(float)*3  + sizeof(float)*3;
    }
    else if(gotxyz && !gotcolor && !gotnormal)
    {
        m_fileType = XYZ;
        m_PointBlockSize = sizeof(float)*3;
    }
    else
    {
        throw std::range_error("Did not find any points in data");
    }
    ifs.close();
}

fileType LineReader::getFileType()
{
    return m_fileType;
}

boost::shared_ptr<void> LineReader::getNextPoints(size_t &return_amount, size_t amount)
{
    FILE * pFile;
    pFile = fopen (m_filePath.c_str(),"r");
    if (pFile!=NULL)
    {
        if(m_ply && m_binary)
        {
            fseek(pFile, m_filePos, SEEK_SET);
            size_t current_pos = ftell(pFile);
            fseek(pFile,0 , SEEK_END);
            size_t last_pos = ftell(pFile);
            size_t data_left = last_pos - current_pos;
            size_t bla;

            data_left = data_left/m_PointBlockSize;
            size_t readSize = amount;
            if(data_left < readSize)
            {
                readSize = data_left;
            }
            fseek(pFile, m_filePos, SEEK_SET);
            boost::shared_ptr<void> pArray(new char[readSize*m_PointBlockSize], std::default_delete<char[]>());
            bla = fread ( pArray.get(), m_PointBlockSize, readSize, pFile );
            fclose (pFile);
            m_filePos += readSize*m_PointBlockSize;
            return_amount = readSize;
            return pArray;
        }
        else
        {
            fseek(pFile, m_filePos, SEEK_SET);
            size_t readCount = 0;
            if(m_fileType == XYZ)
            {
                std::vector<float> input;
                input.reserve(amount*3);
                boost::shared_ptr<void> pArray(new char[amount*m_PointBlockSize], std::default_delete<char[]>());
                float ax, ay, az;
                while( (fscanf(pFile,"%f %f %f", &ax, &ay, &az) != EOF ) && readCount < amount)
                {
                    readCount++;
                    input.push_back(ax);
                    input.push_back(ay);
                    input.push_back(az);
                }
                memcpy ( pArray.get(), input.data(), m_PointBlockSize*readCount );
                return_amount = readCount;
                m_filePos = ftell(pFile);
                return pArray;
            }
            else if(m_fileType == XYZN)
            {
                std::vector<float> input;
                input.reserve(amount*6);
                boost::shared_ptr<void> pArray(new char[amount*m_PointBlockSize], std::default_delete<char[]>());
                float ax, ay, az, nx, ny, nz;
                while( (fscanf(pFile,"%f %f %f %f %f %f", &ax, &ay, &az, &nx, &ny, &nz) != EOF ) && readCount < amount)
                {
                    readCount++;
                    input.push_back(ax);
                    input.push_back(ay);
                    input.push_back(az);
                }
                memcpy ( pArray.get(), input.data(), m_PointBlockSize*readCount );
                return_amount = readCount;
                m_filePos = ftell(pFile);
                return pArray;
            }
            else if(m_fileType == XYZRGB)
            {
                std::vector<xyzc> input;
                input.reserve(amount*6);
                boost::shared_ptr<void> pArray(new char[amount*m_PointBlockSize], std::default_delete<char[]>());
                xyzc pc;
                while( (fscanf(pFile,"%f %f %f %hhu %hhu %hhu", &pc.point.x, &pc.point.y, &pc.point.z, &pc.color.r, &pc.color.g, &pc.color.b) != EOF ) && readCount < amount)
                {
                    readCount++;
                    input.push_back(pc);
                }
                memcpy ( pArray.get(), input.data(), m_PointBlockSize*readCount );
                return_amount = readCount;
                m_filePos = ftell(pFile);
                return pArray;
            }
            else if(m_fileType == XYZNRGB)
            {
                std::vector<xyznc> input;
                input.reserve(amount*6);
                boost::shared_ptr<void> pArray(new char[amount*m_PointBlockSize], std::default_delete<char[]>());
                xyznc pc;
                while( (fscanf(pFile,"%f %f %f %f %f %f %hhu %hhu %hhu",
                               &pc.point.x, &pc.point.y, &pc.point.z,
                               &pc.normal.x, &pc.normal.y, &pc.normal.z,
                               &pc.color.r, &pc.color.g, &pc.color.b) != EOF ) && readCount < amount)
                {
                    readCount++;
                    input.push_back(pc);
                }
                memcpy ( pArray.get(), input.data(), m_PointBlockSize*readCount );
                return_amount = readCount;
                m_filePos = ftell(pFile);
                return pArray;
            }


        }

    }
}

void LineReader::rewind()
{
    open(m_filePath);
}