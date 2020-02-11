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
 * LineReader.cpp
 *
 *  Created on: Aug 15, 2017
 *      Author: Isaak Mitschke
 */

#include <boost/algorithm/string.hpp>
#include <cerrno>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdio.h>

#include "lvr2/io/LineReader.hpp"

namespace lvr2
{

LineReader::LineReader() {}

LineReader::LineReader(std::vector<std::string> filePaths)
    : m_numFiles(1), m_currentReadFile(0), m_openNextFile(false)
{
    open(filePaths);
}

LineReader::LineReader(std::string filePath)
    : m_numFiles(1), m_currentReadFile(0), m_openNextFile(false)
{
    open(filePath);
}

size_t LineReader::getNumPoints()
{
    size_t amount = 0;
    for (size_t i = 0; i < m_fileAttributes.size(); i++)
    {
        amount += m_fileAttributes[i].m_elementAmount;
    }
    return amount;
}

void LineReader::open(std::vector<std::string> filePaths)
{
    m_fileAttributes.clear();
    for (size_t currentFile = 0; currentFile < filePaths.size(); currentFile++)
    {
        fileAttribut currentAttr;
        std::string filePath = filePaths[currentFile];
        currentAttr.m_filePath = filePath;
        bool gotxyz = false;
        bool gotcolor = false;
        bool gotnormal = false;
        bool readHeader = false;

        if (boost::algorithm::contains(filePath, ".ply"))
        {
            // Todo: Check if all files are same type;
            currentAttr.m_ply = true;
        }
        else
        {
            currentAttr.m_ply = false;
        }

        std::ifstream ifs(filePath);

        if (currentAttr.m_ply)
        {
            std::string line;
            while (!readHeader)
            {
                std::getline(ifs, line);
                if (boost::algorithm::contains(line, "element vertex") ||
                    boost::algorithm::contains(line, "element point"))
                {
                    std::stringstream ss(line);
                    string tmp;
                    ss >> tmp;
                    ss >> tmp;
                    ss >> currentAttr.m_elementAmount;
                }
                else if (boost::algorithm::contains(line, "property float x") ||
                         boost::algorithm::contains(line, "property float y") ||
                         boost::algorithm::contains(line, "property float z") ||
                         boost::algorithm::contains(line, "property float32 x") ||
                         boost::algorithm::contains(line, "property float32 y") ||
                         boost::algorithm::contains(line, "property float32 z"))
                {
                    gotxyz = true;
                }
                else if (boost::algorithm::contains(line, "property uchar red") ||
                         boost::algorithm::contains(line, "property uchar green") ||
                         boost::algorithm::contains(line, "property uchar blue"))
                {
                    gotcolor = true;
                }
                else if (boost::algorithm::contains(line, "property float nx") ||
                         boost::algorithm::contains(line, "property float ny") ||
                         boost::algorithm::contains(line, "property float nz"))
                {
                    gotnormal = true;
                }
                else if (boost::algorithm::contains(line, "end_header"))
                {
                    readHeader = true;
                }
                else if (boost::algorithm::contains(line, "binary"))
                {
                    currentAttr.m_binary = true;
                }
                else if (boost::algorithm::contains(line, "ascii"))
                {
                    currentAttr.m_binary = false;
                }
                else if (boost::algorithm::contains(line, "property list"))
                {
                    // Todo...
                }
                else if (boost::algorithm::contains(line, "property"))
                {
                    throw readException((line + " is currently not supported \n supported "
                                                "properties: x y z [red green blue] [nx ny nz]")
                                            .c_str());
                }
            }
            std::cout << "FINISHED READING HEADER" << std::endl;
            std::cout << "XYT:    " << gotxyz << std::endl;
            std::cout << "COLOR:  " << gotcolor << std::endl;
            std::cout << "NORMAL: " << gotnormal << std::endl;
            std::cout << "BINARY: " << currentAttr.m_binary << std::endl;
            std::cout << "Points: " << currentAttr.m_elementAmount << std::endl;
        }
        else
        {
            std::cout << "File Type is not PLY, checking file... " << std::endl;
            std::string line;
            std::getline(ifs, line);
            std::stringstream ss(line);
            string tmp;
            unsigned int number_of_line_elements = 0;
            while (ss >> tmp)
            {
                number_of_line_elements++;
                if (number_of_line_elements >= 3)
                    gotxyz = true;
                if (number_of_line_elements == 6)
                {
                    if (boost::algorithm::contains(tmp, "."))
                    {
                        gotnormal = true;
                    }
                    else
                    {
                        gotcolor = true;
                    }
                }
                if (number_of_line_elements == 9)
                {
                    gotnormal = true;
                    gotcolor = true;
                }
                if (number_of_line_elements > 9)
                {
                    throw std::range_error("Wrong file format, expecting file ascii or ply file "
                                           "format, ascii file format must have order:  x y z [nx "
                                           "ny nz] [cx cy cz] (points, normals, colors)");
                }
            }
            currentAttr.m_line_element_amount = number_of_line_elements;
            ifs.seekg(0);
        }

        currentAttr.m_filePos = ifs.tellg();
        if (gotxyz && gotcolor && gotnormal)
        {
            currentAttr.m_fileType = XYZNRGB;
            currentAttr.m_PointBlockSize =
                sizeof(float) * 3 + sizeof(unsigned char) * 3 + sizeof(float) * 3;
        }
        else if (gotxyz && gotcolor && !gotnormal)
        {
            currentAttr.m_fileType = XYZRGB;
            currentAttr.m_PointBlockSize = sizeof(float) * 3 + sizeof(unsigned char) * 3;
        }
        else if (gotxyz && !gotcolor && gotnormal)
        {
            currentAttr.m_fileType = XYZN;
            currentAttr.m_PointBlockSize = sizeof(float) * 3 + sizeof(float) * 3;
        }
        else if (gotxyz && !gotcolor && !gotnormal)
        {
            currentAttr.m_fileType = XYZ;
            currentAttr.m_PointBlockSize = sizeof(float) * 3;
        }
        else
        {
            throw std::range_error("Did not find any points in data");
        }
        ifs.close();
        m_fileAttributes.push_back(currentAttr);
    }
}

void LineReader::open(std::string filePath)
{
    std::vector<std::string> tmp;
    tmp.push_back(filePath);
    open(tmp);
}

fileType LineReader::getFileType(size_t i)
{
    if (i < m_fileAttributes.size())
    {
        return m_fileAttributes[i].m_fileType;
    }
    else
    {
        throw readException("There is no file with selected index\n (maybe you forgot to rewind "
                            "LineReader when reading file again?)");
    }
}
fileType LineReader::getFileType() { return getFileType(m_currentReadFile); }

bool LineReader::ok() { return m_currentReadFile < m_fileAttributes.size(); }

boost::shared_ptr<void> LineReader::getNextPoints(size_t& return_amount, size_t amount)
{

    return_amount = 0;
    if (m_openNextFile)
    {
        m_openNextFile = false;
        m_currentReadFile++;

        if (m_currentReadFile >= m_fileAttributes.size())
        {
            boost::shared_ptr<void> tmp;
            return tmp;
        }
    }

    std::string filePath = m_fileAttributes[m_currentReadFile].m_filePath;

    FILE* pFile;
    pFile = fopen(filePath.c_str(), "r");
    if (pFile != NULL)
    {
        if (m_fileAttributes[m_currentReadFile].m_ply &&
            m_fileAttributes[m_currentReadFile].m_binary)
        {
            fseek(pFile, m_fileAttributes[m_currentReadFile].m_filePos, SEEK_SET);
            size_t current_pos = ftell(pFile);
            fseek(pFile, 0, SEEK_END);
            size_t last_pos = ftell(pFile);
            size_t data_left = last_pos - current_pos;
            size_t bla;

            data_left = data_left / m_fileAttributes[m_currentReadFile].m_PointBlockSize;
            size_t readSize = amount;
            if (data_left < readSize)
            {
                readSize = data_left;
            }
            fseek(pFile, m_fileAttributes[m_currentReadFile].m_filePos, SEEK_SET);
            boost::shared_ptr<void> pArray(
                new char[readSize * m_fileAttributes[m_currentReadFile].m_PointBlockSize],
                std::default_delete<char[]>());
            bla = fread(pArray.get(),
                        m_fileAttributes[m_currentReadFile].m_PointBlockSize,
                        readSize,
                        pFile);
            fclose(pFile);
            m_fileAttributes[m_currentReadFile].m_filePos +=
                readSize * m_fileAttributes[m_currentReadFile].m_PointBlockSize;
            return_amount = bla;

            if (return_amount < amount)
                m_openNextFile = true;
            return pArray;
        }
        else
        {
            fseek(pFile, m_fileAttributes[m_currentReadFile].m_filePos, SEEK_SET);
            size_t readCount = 0;
            if (m_fileAttributes[m_currentReadFile].m_fileType == XYZ &&
                m_fileAttributes[m_currentReadFile].m_line_element_amount != 3)
            {

                std::vector<float> input;
                input.reserve(amount * 3);
                boost::shared_ptr<void> pArray(
                    new char[amount * m_fileAttributes[m_currentReadFile].m_PointBlockSize],
                    std::default_delete<char[]>());
                float ax, ay, az;
                char lineBuffer[1024];
                while ((fgets(lineBuffer, 1024, pFile) != NULL) && readCount < amount)
                {
                    sscanf(lineBuffer, "%f %f %f", &ax, &ay, &az);
                    readCount++;
                    input.push_back(ax);
                    input.push_back(ay);
                    input.push_back(az);
                }

                memcpy(pArray.get(),
                       input.data(),
                       m_fileAttributes[m_currentReadFile].m_PointBlockSize * readCount);
                return_amount = readCount;
                if (return_amount < amount)
                    m_openNextFile = true;
                m_fileAttributes[m_currentReadFile].m_filePos = ftell(pFile);
                fclose(pFile);
                return pArray;
            }
            if (m_fileAttributes[m_currentReadFile].m_fileType == XYZ)
            {
                std::vector<float> input;
                input.reserve(amount * 3);
                boost::shared_ptr<void> pArray(
                    new char[amount * m_fileAttributes[m_currentReadFile].m_PointBlockSize],
                    std::default_delete<char[]>());
                float ax, ay, az;
                while ((fscanf(pFile, "%f %f %f", &ax, &ay, &az) != EOF) && readCount < amount)
                {
                    readCount++;
                    input.push_back(ax);
                    input.push_back(ay);
                    input.push_back(az);
                }

                memcpy(pArray.get(),
                       input.data(),
                       m_fileAttributes[m_currentReadFile].m_PointBlockSize * readCount);
                return_amount = readCount;
                if (return_amount < amount)
                {
                    m_openNextFile = true;
                }
                else
                {
                    m_openNextFile = false;
                }
                m_fileAttributes[m_currentReadFile].m_filePos = ftell(pFile);
                fclose(pFile);
                return pArray;
            }
            else if (m_fileAttributes[m_currentReadFile].m_fileType == XYZN)
            {
                std::vector<float> input;
                input.reserve(amount * 6);
                boost::shared_ptr<void> pArray(
                    new char[amount * m_fileAttributes[m_currentReadFile].m_PointBlockSize],
                    std::default_delete<char[]>());
                float ax, ay, az, nx, ny, nz;
                while ((fscanf(pFile, "%f %f %f %f %f %f", &ax, &ay, &az, &nx, &ny, &nz) != EOF) &&
                       readCount < amount)
                {
                    readCount++;
                    input.push_back(ax);
                    input.push_back(ay);
                    input.push_back(az);
                }
                memcpy(pArray.get(),
                       input.data(),
                       m_fileAttributes[m_currentReadFile].m_PointBlockSize * readCount);
                return_amount = readCount;
                if (return_amount < amount)
                    m_openNextFile = true;
                m_fileAttributes[m_currentReadFile].m_filePos = ftell(pFile);
                fclose(pFile);
                return pArray;
            }
            else if (m_fileAttributes[m_currentReadFile].m_fileType == XYZRGB)
            {
                std::vector<xyzc> input;
                input.reserve(amount * 6);
                boost::shared_ptr<void> pArray(
                    new char[amount * m_fileAttributes[m_currentReadFile].m_PointBlockSize],
                    std::default_delete<char[]>());
                xyzc pc;
                float tmp_x, tmp_y, tmp_z;
                unsigned char tmp_r, tmp_g, tmp_b;
                while ((fscanf(pFile,
                               "%f %f %f %hhu %hhu %hhu",  // don't read directly in to struct!
                               &tmp_x,
                               &tmp_y,
                               &tmp_z,
                               &tmp_r,
                               &tmp_g,
                               &tmp_b) != EOF) &&
                       readCount < amount)
                {
                    pc.point.x = tmp_x;
                    pc.point.y = tmp_y;
                    pc.point.z = tmp_z;
                    pc.color.r = tmp_r;
                    pc.color.g = tmp_g;
                    pc.color.b = tmp_b;
                    readCount++;
                    input.push_back(pc);
                }
                memcpy(pArray.get(),
                       input.data(),
                       m_fileAttributes[m_currentReadFile].m_PointBlockSize * readCount);
                return_amount = readCount;
                if (return_amount < amount)
                    m_openNextFile = true;
                m_fileAttributes[m_currentReadFile].m_filePos = ftell(pFile);
                fclose(pFile);
                return pArray;
            }
            else if (m_fileAttributes[m_currentReadFile].m_fileType == XYZNRGB)
            {
                std::vector<xyznc> input;
                input.reserve(amount * 6);
                boost::shared_ptr<void> pArray(
                    new char[amount * m_fileAttributes[m_currentReadFile].m_PointBlockSize],
                    std::default_delete<char[]>());
                xyznc pc;
                float tmp_x, tmp_y, tmp_z, tmp_nx, tmp_ny, tmp_nz;
                unsigned char tmp_r, tmp_g, tmp_b;
                while ((fscanf(pFile,
                               "%f %f %f %hhu %hhu %hhu %f %f %f",
                               &tmp_x,
                               &tmp_y,
                               &tmp_z,
                               &tmp_r,
                               &tmp_g,
                               &tmp_b,
                               &tmp_nx,
                               &tmp_ny,
                               &tmp_nz) != EOF) &&
                       readCount < amount)
                {
                    pc.point.x = tmp_x;
                    pc.point.y = tmp_y;
                    pc.point.z = tmp_z;
                    pc.color.r = tmp_r;
                    pc.color.g = tmp_g;
                    pc.color.b = tmp_b;
                    pc.normal.x = tmp_nx;
                    pc.normal.y = tmp_ny;
                    pc.normal.z = tmp_nz;
                    readCount++;
                    input.push_back(pc);
                }
                memcpy(pArray.get(),
                       input.data(),
                       m_fileAttributes[m_currentReadFile].m_PointBlockSize * readCount);
                return_amount = readCount;
                m_fileAttributes[m_currentReadFile].m_filePos = ftell(pFile);
                if (return_amount < amount)
                {
                    m_openNextFile = true;
                }
                fclose(pFile);
                return pArray;
            }
        }
        fclose(pFile);
    }
    else
    {
        std::cout << "SHIT could not open file: " << std::strerror(errno) << std::endl;
    }

    // Return empty pointer if all else fails...
    boost::shared_ptr<void> tmp;
    return tmp;
}

void LineReader::rewind()
{
    std::vector<std::string> tmp;
    for (size_t i = 0; i < m_fileAttributes.size(); i++)
    {
        tmp.push_back(m_fileAttributes[i].m_filePath);
    }
    open(tmp);
    m_currentReadFile = 0;
}

void LineReader::rewind(size_t i) { open(m_fileAttributes[i].m_filePath); }

} // namespace lvr2
