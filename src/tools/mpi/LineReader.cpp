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

LineReader::LineReader(std::vector<std::string> filePaths) : m_numFiles(1), m_currentReadFile(0), m_openNextFile(false)
{
    open(filePaths);
}

LineReader::LineReader(std::string filePath) : m_numFiles(1), m_currentReadFile(0), m_openNextFile(false)
{
    open(filePath);
}

size_t LineReader::getNumPoints() {
    size_t amount = 0;
    for(size_t i = 0 ; i<m_fileAttributes.size();i++)
    {
        amount +=m_fileAttributes[i].m_elementAmount;
    }
    return amount;
}

void LineReader::open(std::vector<std::string> filePaths)
{
    m_fileAttributes.clear();
    for(size_t currentFile = 0 ; currentFile < filePaths.size() ; currentFile++)
    {
        fileAttribut currentAttr;
        std::string filePath = filePaths[currentFile];
        currentAttr.m_filePath = filePath;
        bool gotxyz = false;
        bool gotcolor = false;
        bool gotnormal = false;
        bool readHeader = false;

        if(boost::algorithm::contains(filePath,".ply"))
        {
            //Todo: Check if all files are same type;
            currentAttr.m_ply = true;
        }
        else
        {
            currentAttr.m_ply = false;
        }

        std::ifstream ifs(filePath);

        if(currentAttr.m_ply)
        {
            std::string line;
            while(! readHeader)
            {
                std::getline(ifs,line);
                if(boost::algorithm::contains(line,"element vertex") || boost::algorithm::contains(line,"element point"))
                {
                    std::stringstream ss(line);
                    string tmp;
                    ss >> tmp;
                    ss >> tmp;
                    ss >> currentAttr.m_elementAmount;
                }
                else if(boost::algorithm::contains(line,"property float x") ||
                        boost::algorithm::contains(line,"property float y") ||
                        boost::algorithm::contains(line,"property float z") ||
                        boost::algorithm::contains(line,"property float32 x") ||
                        boost::algorithm::contains(line,"property float32 y") ||
                        boost::algorithm::contains(line,"property float32 z")
                        )
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
                    currentAttr.m_binary = true;
                }
                else if(boost::algorithm::contains(line,"ascii"))
                {
                    currentAttr.m_binary = false;
                }
                else if(boost::algorithm::contains(line,"property list"))
                {
                    //Todo...
                }
                else if(boost::algorithm::contains(line,"property"))
                {
                    throw readException((line + " is currently not supported \n supported properties: x y z [red green blue] [nx ny nz]").c_str());
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
            std::getline(ifs,line);
            std::stringstream ss(line);
            string tmp;
            unsigned int number_of_line_elements = 0;
            while(ss >> tmp)
            {
                number_of_line_elements++;
                if(number_of_line_elements >= 3) gotxyz = true;
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
            currentAttr.m_line_element_amount = number_of_line_elements;
            ifs.seekg(0);


        }

        currentAttr.m_filePos = ifs.tellg();
        if(gotxyz && gotcolor && gotnormal)
        {
            currentAttr.m_fileType = XYZNRGB;
            currentAttr.m_PointBlockSize = sizeof(float)*3 + sizeof(unsigned char) * 3 + sizeof(float)*3;
        }
        else if(gotxyz && gotcolor && !gotnormal)
        {
            currentAttr.m_fileType = XYZRGB;
            currentAttr.m_PointBlockSize = sizeof(float)*3 + sizeof(unsigned char) * 3;
        }
        else if(gotxyz && !gotcolor && gotnormal)
        {
            currentAttr.m_fileType = XYZN;
            currentAttr.m_PointBlockSize = sizeof(float)*3  + sizeof(float)*3;
        }
        else if(gotxyz && !gotcolor && !gotnormal)
        {
            currentAttr.m_fileType = XYZ;
            currentAttr.m_PointBlockSize = sizeof(float)*3;
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
    if(i < m_fileAttributes.size())
    {
        return m_fileAttributes[i].m_fileType;
    }
    else
    {
        throw readException("There is no file with selected index\n (maybe you forgot to rewind LineReader when reading file again?)");
    }
}
fileType LineReader::getFileType()
{
   getFileType(m_currentReadFile);
}

bool LineReader::ok()
{
    return m_currentReadFile < m_fileAttributes.size();
}

boost::shared_ptr<void> LineReader::getNextPoints(size_t &return_amount, size_t amount)
{
    return_amount = 0;
    if(m_openNextFile)
    {
        m_openNextFile = false;
        m_currentReadFile++;
        if(m_currentReadFile>=m_fileAttributes.size())
        {
            boost::shared_ptr<void> tmp;
            return tmp;
        }
    }

    std::string filePath = m_fileAttributes[m_currentReadFile].m_filePath;
    
    FILE * pFile;
    pFile = fopen(filePath.c_str(), "r");
    if (pFile!=NULL)
    {
        if(m_fileAttributes[m_currentReadFile].m_ply && m_fileAttributes[m_currentReadFile].m_binary)
        {
            fseek(pFile, m_fileAttributes[m_currentReadFile].m_filePos, SEEK_SET);
            size_t current_pos = ftell(pFile);
            fseek(pFile,0 , SEEK_END);
            size_t last_pos = ftell(pFile);
            size_t data_left = last_pos - current_pos;
            size_t bla;

            data_left = data_left/m_fileAttributes[m_currentReadFile].m_PointBlockSize;
            size_t readSize = amount;
            if(data_left < readSize)
            {
                readSize = data_left;
            }
            fseek(pFile, m_fileAttributes[m_currentReadFile].m_filePos, SEEK_SET);
            boost::shared_ptr<void> pArray(new char[readSize*m_fileAttributes[m_currentReadFile].m_PointBlockSize], std::default_delete<char[]>());
            bla = fread ( pArray.get(), m_fileAttributes[m_currentReadFile].m_PointBlockSize, readSize, pFile );
            fclose (pFile);
            m_fileAttributes[m_currentReadFile].m_filePos += readSize*m_fileAttributes[m_currentReadFile].m_PointBlockSize;
            return_amount = readSize;
            if(return_amount < amount) m_openNextFile = true;
            return pArray;
        }
        else
        {
            fseek(pFile, m_fileAttributes[m_currentReadFile].m_filePos, SEEK_SET);
            size_t readCount = 0;
            if(m_fileAttributes[m_currentReadFile].m_fileType == XYZ && m_fileAttributes[m_currentReadFile].m_line_element_amount != 3)
            {
                
                std::vector<float> input;
                input.reserve(amount*3);
                boost::shared_ptr<void> pArray(new char[amount*m_fileAttributes[m_currentReadFile].m_PointBlockSize], std::default_delete<char[]>());
                float ax, ay, az;
                char lineBuffer[1024];
                while((fgets( lineBuffer, 1024, pFile ) != NULL)  && readCount < amount)
                {
                    sscanf(lineBuffer,"%f %f %f", &ax, &ay, &az);
                    readCount++;
                    input.push_back(ax);
                    input.push_back(ay);
                    input.push_back(az);
                }

                memcpy ( pArray.get(), input.data(), m_fileAttributes[m_currentReadFile].m_PointBlockSize*readCount );
                return_amount = readCount;
                if(return_amount < amount) m_openNextFile = true;
                m_fileAttributes[m_currentReadFile].m_filePos = ftell(pFile);
                fclose(pFile);
                return pArray;
            }
            if(m_fileAttributes[m_currentReadFile].m_fileType == XYZ)
            {
                std::vector<float> input;
                input.reserve(amount*3);
                boost::shared_ptr<void> pArray(new char[amount*m_fileAttributes[m_currentReadFile].m_PointBlockSize], std::default_delete<char[]>());
                float ax, ay, az;
                while( (fscanf(pFile,"%f %f %f", &ax, &ay, &az) != EOF ) && readCount < amount )
                {
                    readCount++;
                    input.push_back(ax);
                    input.push_back(ay);
                    input.push_back(az);
                }
                
                memcpy ( pArray.get(), input.data(), m_fileAttributes[m_currentReadFile].m_PointBlockSize*readCount );
                return_amount = readCount;
                if(return_amount < amount){
                    m_openNextFile = true;
                } else{
                    m_openNextFile = false;
                }
                m_fileAttributes[m_currentReadFile].m_filePos = ftell(pFile);
                fclose(pFile);
                return pArray;
            }
            else if(m_fileAttributes[m_currentReadFile].m_fileType == XYZN)
            {
                std::vector<float> input;
                input.reserve(amount*6);
                boost::shared_ptr<void> pArray(new char[amount*m_fileAttributes[m_currentReadFile].m_PointBlockSize], std::default_delete<char[]>());
                float ax, ay, az, nx, ny, nz;
                while( (fscanf(pFile,"%f %f %f %f %f %f", &ax, &ay, &az, &nx, &ny, &nz) != EOF ) && readCount < amount)
                {
                    readCount++;
                    input.push_back(ax);
                    input.push_back(ay);
                    input.push_back(az);
                }
                memcpy ( pArray.get(), input.data(), m_fileAttributes[m_currentReadFile].m_PointBlockSize*readCount );
                return_amount = readCount;
                if(return_amount < amount) m_openNextFile = true;
                m_fileAttributes[m_currentReadFile].m_filePos = ftell(pFile);
                fclose(pFile);
                return pArray;
            }
            else if(m_fileAttributes[m_currentReadFile].m_fileType == XYZRGB)
            {
                std::vector<xyzc> input;
                input.reserve(amount*6);
                boost::shared_ptr<void> pArray(new char[amount*m_fileAttributes[m_currentReadFile].m_PointBlockSize], std::default_delete<char[]>());
                xyzc pc;
                while( (fscanf(pFile,"%f %f %f %hhu %hhu %hhu", &pc.point.x, &pc.point.y, &pc.point.z, &pc.color.r, &pc.color.g, &pc.color.b) != EOF ) && readCount < amount)
                {
                    readCount++;
                    input.push_back(pc);
                }
                memcpy ( pArray.get(), input.data(), m_fileAttributes[m_currentReadFile].m_PointBlockSize*readCount );
                return_amount = readCount;
                if(return_amount < amount) m_openNextFile = true;
                m_fileAttributes[m_currentReadFile].m_filePos = ftell(pFile);
                fclose(pFile);
                return pArray;
            }
            else if(m_fileAttributes[m_currentReadFile].m_fileType == XYZNRGB)
            {
                std::vector<xyznc> input;
                input.reserve(amount*6);
                boost::shared_ptr<void> pArray(new char[amount*m_fileAttributes[m_currentReadFile].m_PointBlockSize], std::default_delete<char[]>());
                xyznc pc;
                while( (fscanf(pFile,"%f %f %f %hhu %hhu %hhu %f %f %f",
                               &pc.point.x, &pc.point.y, &pc.point.z,
                               &pc.color.r, &pc.color.g, &pc.color.b,
                               &pc.normal.x, &pc.normal.y, &pc.normal.z) != EOF ) && readCount < amount)
                {
                    readCount++;
                    input.push_back(pc);
                }
                memcpy ( pArray.get(), input.data(), m_fileAttributes[m_currentReadFile].m_PointBlockSize*readCount );
                return_amount = readCount;
                m_fileAttributes[m_currentReadFile].m_filePos = ftell(pFile);
                if(return_amount < amount)
                {
                    m_openNextFile = true;
                }
                fclose(pFile);
                return pArray;
            }


        }
        fclose(pFile);
    }
    
    

}
void LineReader::rewind()
{
    std::vector<std::string> tmp;
    for(size_t i = 0 ; i< m_fileAttributes.size();i++)
    {
        tmp.push_back(m_fileAttributes[i].m_filePath);
    }
    open(tmp);
    m_currentReadFile = 0;
}
void LineReader::rewind(size_t i)
{
    open(m_fileAttributes[i].m_filePath);

}
