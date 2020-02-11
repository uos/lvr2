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
 * NodeData.cpp
 *
 *  Created on: Dec 11, 2015
 *      Author: Isaak Mitschke
 */

#include "lvr2/reconstruction/NodeData.hpp"

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <sstream>
#include <vector>

using namespace std;
namespace lvr2
{

template <typename BaseVecT>
boost::timer::cpu_timer NodeData<BaseVecT>::itimer;

template <typename BaseVecT>
boost::timer::cpu_timer NodeData<BaseVecT>::otimer;

template <typename BaseVecT>
bool NodeData<BaseVecT>::timer_init = false;

template <typename BaseVecT>
int NodeData<BaseVecT>::c_last_id = 0;

template <typename BaseVecT>
time_t NodeData<BaseVecT>::c_tstamp = std::time(0);

template <typename BaseVecT>
NodeData<BaseVecT>::NodeData(size_t bufferSize) : m_bufferSize(bufferSize)
{

    if (!timer_init)
    {
        itimer.stop();
        otimer.stop();
        timer_init = true;
    }
    m_gotSize = false;
    m_id = ++c_last_id;
    m_dataPath = "node-";
    m_dataPath.append(to_string(c_tstamp));
    m_dataPath.append("/");
    boost::filesystem::path dir(m_dataPath);

    if (!(boost::filesystem::exists(dir)))
    {
        boost::filesystem::create_directory(dir);
    }
    m_dataPath.append(to_string(m_id));
    m_dataPathNormal = m_dataPath;
    m_dataPath.append(".xyz");
    m_dataPathNormal.append(".normals");
    m_readBufferIndex = 0;
    m_readBufferIndexNormal = 0;
    m_writeBuffer.clear();
    m_writeBufferNormal.clear();
    vector<float>().swap(m_writeBuffer);
    vector<float>().swap(m_writeBufferNormal);
}

template <typename BaseVecT>
void NodeData<BaseVecT>::fillBuffer(size_t start_id)
{
    m_readBuffer.clear();
    m_readBuffer.resize(m_bufferSize);
    FILE* pfile = fopen(m_dataPath.c_str(), "rb");
    m_readBufferIndex = start_id;
    itimer.resume();
    fseek(pfile, sizeof(float) * start_id * 3, SEEK_SET);
    itimer.stop();
    size_t readamount = fread(m_readBuffer.data(), sizeof(float), m_bufferSize, pfile);
    fclose(pfile);
    m_readBuffer.resize(readamount);
}

template <typename BaseVecT>
void NodeData<BaseVecT>::fillBufferNormal(size_t start_id)
{
    m_readBufferNormal.clear();
    m_readBufferNormal.resize(m_bufferSize);
    FILE* pfilen = fopen(m_dataPathNormal.c_str(), "rb");
    m_readBufferIndexNormal = start_id;
    itimer.resume();
    fseek(pfilen, sizeof(float) * start_id * 3, SEEK_SET);
    itimer.stop();
    size_t readamount = fread(m_readBufferNormal.data(), sizeof(float), m_bufferSize, pfilen);
    fclose(pfilen);
    m_readBufferNormal.resize(readamount);
}

template <typename BaseVecT>
void NodeData<BaseVecT>::open(string path)
{
    m_dataPath = path;
}

template <typename BaseVecT>
void NodeData<BaseVecT>::remove()
{
    boost::filesystem::remove(m_dataPath);
    boost::filesystem::remove(m_dataPathNormal);
    m_dataPath = "";
    m_dataPathNormal = "";
    m_readBuffer.clear();
    m_readBufferNormal.clear();
    vector<float>().swap(m_readBuffer);
    vector<float>().swap(m_readBufferNormal);
}

template <typename BaseVecT>
void NodeData<BaseVecT>::remove(unsigned int i)
{
    // Todo: remove vertex at index
}

template <typename BaseVecT>
void NodeData<BaseVecT>::add(BaseVecT input)
{
    FILE* oFile = fopen(m_dataPath.c_str(), "ab");
    float v[3];
    v[0] = input[0];
    v[1] = input[1];
    v[2] = input[2];
    otimer.resume();
    fwrite(m_writeBuffer.data(), sizeof(float), 3, oFile);
    otimer.stop();
    fclose(oFile);

    //    ofstream ofs(m_dataPath, fstream::app);
    //    ofs << input.x << " " << input.y << " " <<  input.z << " " <<  std::endl;
    //    if(m_gotSize) m_size++;
    //    ofs.close();
}

template <typename BaseVecT>
void NodeData<BaseVecT>::addNormal(BaseVecT input)
{
    FILE* oFile = fopen(m_dataPathNormal.c_str(), "ab");
    float v[3];
    v[0] = input[0];
    v[1] = input[1];
    v[2] = input[2];
    otimer.resume();
    fwrite(m_writeBufferNormal.data(), sizeof(float), 3, oFile);
    otimer.stop();
    fclose(oFile);

    //    ofstream ofs(m_dataPath, fstream::app);
    //    ofs << input.x << " " << input.y << " " <<  input.z << " " <<  std::endl;
    //    if(m_gotSize) m_size++;
    //    ofs.close();
}

template <typename BaseVecT>
void NodeData<BaseVecT>::addBuffered(BaseVecT input)
{

    if (m_gotSize)
        m_size++;
    if (m_writeBuffer.size() > m_bufferSize)
        writeBuffer();
    m_writeBuffer.push_back(input.x);
    m_writeBuffer.push_back(input.y);
    m_writeBuffer.push_back(input.z);
}

template <typename BaseVecT>
void NodeData<BaseVecT>::addBufferedNormal(BaseVecT input)
{
    if (m_writeBufferNormal.size() > m_bufferSize)
        writeBuffer();
    m_writeBufferNormal.push_back(input.x);
    m_writeBufferNormal.push_back(input.y);
    m_writeBufferNormal.push_back(input.z);
}

template <typename BaseVecT>
void NodeData<BaseVecT>::writeBuffer()
{

    if (m_writeBuffer.size() == 0)
        return;
    FILE* oFile = fopen(m_dataPath.c_str(), "a+b");
    if (oFile != NULL)
    {
        otimer.resume();
        fwrite(m_writeBuffer.data(), sizeof(float), m_writeBuffer.size(), oFile);
        otimer.stop();
        fclose(oFile);
    }
    else
    {
        cout << "ERROR: " << errno << ": " << strerror(errno) << endl;
        throw std::runtime_error("asd");
    }

    m_writeBuffer.clear();
    vector<float>().swap(m_writeBuffer);
    // Normals
    if (m_writeBufferNormal.size() == 0)
        return;
    oFile = fopen(m_dataPathNormal.c_str(), "a+b");
    if (oFile != NULL)
    {
        otimer.resume();
        fwrite(m_writeBufferNormal.data(), sizeof(float), m_writeBufferNormal.size(), oFile);
        otimer.stop();
        fclose(oFile);
    }
    else
    {
        cout << "ERROR: " << errno << ": " << strerror(errno) << endl;
        throw std::runtime_error("asd");
    }

    m_writeBufferNormal.clear();
    vector<float>().swap(m_writeBufferNormal);
}

template <typename BaseVecT>
size_t NodeData<BaseVecT>::getMaxWriteBufferSize()
{
    return m_bufferSize;
}

template <typename BaseVecT>
size_t NodeData<BaseVecT>::getBufferSize()
{
    return m_writeBuffer.size();
}

template <typename BaseVecT>
BaseVecT NodeData<BaseVecT>::get(int i)
{

    //    cout << "s " << m_readBuffer.size() << endl;
    if (i >= m_readBufferIndex && i - m_readBufferIndex < m_readBuffer.size() / 3)
    {
        //        cout << "got data from buffer" << endl;
        BaseVecT ret(m_readBuffer[(i - m_readBufferIndex) * 3],
                     m_readBuffer[((i - m_readBufferIndex) * 3) + 1],
                     m_readBuffer[((i - m_readBufferIndex) * 3) + 2]);
        return ret;
    }
    else
    {
        //        cout << "read buffer again" << endl;
        fillBuffer(i);

        BaseVecT ret(m_readBuffer[(i - m_readBufferIndex) * 3],
                     m_readBuffer[((i - m_readBufferIndex) * 3) + 1],
                     m_readBuffer[((i - m_readBufferIndex) * 3) + 2]);
        return ret;
    }
}

template <typename BaseVecT>
BaseVecT NodeData<BaseVecT>::getNormal(int i)
{

    //    cout << "s " << m_readBuffer.size() << endl;
    if (i >= m_readBufferIndexNormal && i - m_readBufferIndexNormal < m_readBufferNormal.size() / 3)
    {
        //        cout << "got data from buffer" << endl;
        BaseVecT ret(m_readBufferNormal[(i - m_readBufferIndexNormal) * 3],
                     m_readBufferNormal[((i - m_readBufferIndexNormal) * 3) + 1],
                     m_readBufferNormal[((i - m_readBufferIndexNormal) * 3) + 2]);
        return ret;
    }
    else
    {
        //        cout << "read buffer again" << endl;
        fillBufferNormal(i);

        BaseVecT ret(m_readBufferNormal[(i - m_readBufferIndexNormal) * 3],
                     m_readBufferNormal[((i - m_readBufferIndexNormal) * 3) + 1],
                     m_readBufferNormal[((i - m_readBufferIndexNormal) * 3) + 2]);
        return ret;
    }
}

template <typename BaseVecT>
BaseVecT NodeData<BaseVecT>::next()
{
    return BaseVecT();
}

template <typename BaseVecT>
typename NodeData<BaseVecT>::Iterator NodeData<BaseVecT>::begin()
{
    return NodeData::Iterator(*this);
}

template <typename BaseVecT>
typename NodeData<BaseVecT>::Iterator NodeData<BaseVecT>::end()
{
    return NodeData::Iterator(*this, this->size());
}

template <typename BaseVecT>
NodeData<BaseVecT>::NodeData(NodeData& origin) : NodeData(origin.m_bufferSize)
{
    copy(origin);
}

template <typename BaseVecT>
NodeData<BaseVecT>& lvr2::NodeData<BaseVecT>::operator=(lvr2::NodeData<BaseVecT>& origin)
{
    if (this != &origin)
    {
        copy(origin);
    }
    return *this;
}

/*template <typename BaseVecT>
BaseVecT lvr2::NodeData<BaseVecT>::operator[](unsigned int i)
{
    return get(i);
}*/

template <typename BaseVecT>
void NodeData<BaseVecT>::copy(NodeData& origin)
{
    this->m_dataPath = origin.m_dataPath;
    this->m_gotSize = origin.m_gotSize;
    this->m_size = origin.m_size;
}

template <typename BaseVecT>
size_t NodeData<BaseVecT>::size()
{
    // writeBuffer();
    if (m_gotSize)
        return m_size;
    else
    {

        FILE* fp = fopen(m_dataPath.c_str(), "rb");
        size_t sz = 0;
        if (fp != NULL)
        {
            itimer.resume();
            fseek(fp, 0L, SEEK_END);
            sz = ftell(fp);
            itimer.stop();
            fclose(fp);
            sz /= sizeof(float);
            sz /= 3;
        }

        //        ifstream ifs( m_dataPath );
        //        size_t size = 0;
        //        string s;
        //        while( getline( ifs, s ) ) size++;

        m_size = sz;
        m_gotSize = true;
        //        ifs.close();
        sz += (m_writeBuffer.size() / 3);
        return sz;
    }
}

} // namespace lvr2
